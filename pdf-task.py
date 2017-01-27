from __future__ import (unicode_literals, absolute_import)


import logging
import pysolr
import requests
import tempfile

from celery import states
from datetime import datetime
from hashlib import sha512

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import *
from pdfminer.pdftypes import PDFException

from degreedata.settings.base import S3, PDF_SOLR as SOLR, PDF_PAGE_SOLR
from ingest.models import PdfValidateCatalog
from index.models import PdfIndexDocument

from . import AbstractIndexTask, AbstractIndexVerifyTask


class PdfElementIndexer(AbstractIndexTask):
    def __init__(self):
        self.log = logging.getLogger('django')
        self.limit = None
        self.catalogs = None
        self.aws_url = '{0}/{1}/{2}.pdf'.format(S3['endpoint'], S3['pdf_bucket'], '{0}')
        self._resource_mgr = PDFResourceManager()
        self._layout_params = LAParams()
        self.parser = None
        self.temp = None

    def _fetch_document(self, cl_id):
        url = self.aws_url.format(cl_id)
        self.log.debug('Retrieving PDF URL [{0}]'.format(url))

        req = requests.get(url, stream = True)
        if req.status_code >= 400:
            raise requests.RequestException('URL [{0}] return status [{1}]'.format(url, req.status_code))

        req.raw.decode_content = True
        self.temp = tempfile.TemporaryFile()
        self.temp.write(req.raw.data)
        self.temp.seek(0)
        self.log.info('Successful Retrieval and temporary file creation.  Initializing PDF Extraction')

        self.log.info('Initializing PDF Parser')
        self.parser = PDFParser(self.temp)

        self.log.info('Initializing PDF Document')
        doc = PDFDocument(self.parser)

        self.log.info('Linking Document and Parser')
        self.parser.set_document(doc)

        req.close()

        return doc

    def parse_lt_objects(self, layout, index, text = []):
        """
        Iterates over a list of LT* objects and captures the text contained within,  Images are skipped
        :param layout: List of LT* objects retrieved from the PDFPage instance
        :param index: Current page number
        :return: String of text
        """

        self.log.debug('Processing LT objects for page [{0}]'.format(index))
        text_content = []
        page_text = {}  # k = (x0, x1) of bounding box, v = list of text strings within that column

        for obj in layout:
            if isinstance(obj, LTTextBox) or isinstance(obj, LTTextLine):
                self.log.debug('[{0}] object found'.format(type(obj)))
                page_text = self._update_text_hash(page_text, obj)
            elif isinstance(obj, LTFigure):
                # LTFigure objects are containers for other LT* objects, so recurse through children
                self.log.debug('LTFigure object found, recursing to process children nodes')
                text_content.append(self.parse_lt_objects(obj, index, text_content))

        self.log.debug('Page [{0}] extracted'.format(index))
        return page_text

    def _update_text_hash(self, text, obj, pct = 0.2):
        """
            Use the bbox x0, x1 values within :param pct to produce lists of associated text within the hash

            :param text: dict of page text in the format {(x0, x1) : [list of strings in that column]
            :param lt_obj: LineText object
            :return: hash of text values mapped to bounding boxes
            """

        x_0 = obj.bbox[0]
        x_1 = obj.bbox[1]

        key_found = False
        self.log.debug('Updating page text hash for bbox [({0}, {1})]'.format(x_0, x_1))

        for k, v in text.items() :
            hash_x0 = k[0]
            if x_0 >= (hash_x0 * (1.0 - pct)) and (hash_x0 * (1.0 + pct)) >= x_0 :
                hash_x1 = k[1]
                if x_1 >= (hash_x1 * (1.0 - pct)) and (hash_x1 * (1.0 + pct)) >= x_1 :
                    # text inside this LT object was positioned at the same width as a prior series of text, so it
                    # belongs together
                    key_found = True
                    v.append(self._remove_non_ascii(obj.get_text()))
                    text[k] = v
                    self.log.debug('BBox [{0}, {1}] text updated'.format(x_0, x_1))

        if not key_found :
            # Based on width of bounding box, this text is a new series, so it gets its own entry in the hash
            text[(x_0, x_1)] = [self._remove_non_ascii(obj.get_text())]
            self.log.debug('Created new hash key for bbox [{0}, {1}]'.format(x_0, x_1))

        return text

    def _parse_pages(self, document):
        """
        With an open PDFDocument object, get the pages and parse each one.  This is a higher order function to be
        passed in the run() method as the fn parameter
        :param document: PDFDocument object
        :return: list of text extracted
        """
        self.log.info('Initializing Page Aggregator')
        device = PDFPageAggregator(self._resource_mgr, laparams = self._layout_params)

        self.log.info('Initializing Page Interpreter')
        interpreter = PDFPageInterpreter(self._resource_mgr, device)

        text_content = []

        for idx, page in enumerate(PDFPage.create_pages(document)):
            self.log.debug('Interpreter processing page [{0}]'.format(idx))
            interpreter.process_page(page)

            self.log.debug('Retrieved LTPage object for page')
            layout = device.get_result()

            text_content.append(self.parse_lt_objects(layout, idx))

        self.log.info('Successfully completed text extraction of [{0}] pages'.format(len(text_content)))
        return text_content

    def _to_bytestring(self, string, encode = 'utf-8'):
        """
        Convert a given unicode string to a byte string, using standard encoding.
        :param string: Unicode string
        :param encode: Encoding Format
        :return: bytestring encoded in :param encode forma
        """

        if string :
            if isinstance(string, str):
                return string
            else:
                return string.encode(encode)

    def _remove_non_ascii(self, s):
        # Project uses Python 2.7, which comes with a host of unicode issues.  This attempts to sidestep, as we are
        # concentrating only on English-language documents.
        try:
            return u"".join(
                    i for i in s if ord(i) < 128 and (ord(i) >= 32 or ord(i) == 9 or ord(i) == 10 or ord(i) == 13))
        except Exception:
            return ""

    def _close(self):
        if self.parser:
            self.parser.close()
        if self.temp:
            self.temp.close()

    def _save_state(self, cl_id, pdf_validate_status, index_status, documents_indexed = 0, index_message = ''):
        # We need data in the main application MySQL db updated to reflect the text extraction and indexing status
        validate_catalog, created = PdfValidateCatalog.objects.get_or_create(catalog__link_id = cl_id)

        validate_catalog.index_status = index_status
        validate_catalog.message = index_message
        validate_catalog.documents_indexed = documents_indexed
        validate_catalog.save()
        self._close()
        self.log.info('Saving state with message: [{0}]'.format(index_message))

        return validate_catalog

    def _save_to_db(self, data, cl_id, catalog_year, institution):
        # One of two save methods, saves to a relational database configured and optimized for text search
        self.log.info('Starting saving data to database')

        document_list = []
        indexed_date = datetime.now().strftime('%c')
        page_count = 0

        for idx, val in enumerate(data):
            if val != '':
                page_count += 1
                self.log.info('Indexing page [{0}]'.format(idx))

                for k, v in val.iteritems():
                    section_text = '\n'.join(v)
                    section_id = sha512(section_text).hexdigest()
                    entry = PdfIndexDocument(hash_id = section_id, page = page_count, bounds = repr(k),
                                             content = section_text.decode('utf-8'), catalog_link = cl_id,
                                             catalog_year = catalog_year, institution = institution,
                                             indexed_date = indexed_date)

                    document_list.append(entry)

        PdfIndexDocument.objects.bulk_create(document_list)
        return len(document_list)

    def _solr(self, data, cl_id, catalog_year, institution, soft_commit = True):
        # The second of two save methods, saves to a Solr server
        self.log.info('Starting SOLR indexing with instance URL [{0}]'.format(SOLR))
        solr_instance = pysolr.Solr(SOLR, timeout = 360)
        page_count = 0
        solr_data = []
        indexed_date = datetime.now().strftime('%c')

        for idx, val in enumerate(data):
            if val != '':
                page_count += 1
                self.log.debug('Indexing page [{0}]'.format(idx))
                for k, v in val.iteritems():

                    if type(k) is tuple:  # Ensure key is always tuple to be iterated over
                        section_text = '\n'.join(v)
                        section_id = sha512(section_text).hexdigest()
                        solr_data.append({
                            'id': section_id,
                            'page': page_count,
                            'bounds': repr(k),
                            'content': section_text.decode('utf-8'),
                            'catalog_link': cl_id,
                            'catalog_year': catalog_year,
                            'institution': institution,
                            'indexed_date': indexed_date
                        })

        self.log.info('Committing [{0}] pages of content'.format(page_count))
        solr_instance.delete(q='catalog_link:{0}'.format(cl_id))
        solr_instance.add(solr_data, waitSearcher = True)

        return len(solr_data)

    def on_failure(self, exc, task_id, args, kwargs, einfo) :
        self.log.error('Error for task [{0}] in indexing PDF document [{1}]'.format(task_id, args[0]))
        self.log.error('Einfo: [{0}]'.format(einfo))

    def run(self, cl_id, catalog_year, institution, db_insert = False, soft_commit = True):
        """
        Main run method for this Celery task.  For a provided :param cl_id, the associated PDF file will be retrieved from
        S3 for text extraction and insert to the SOLR server for search and data retrieval.

        :param cl_id: CatalogLink ID for PDF document to be indexed
        :param catalog_year String for catalog year
        :param institution String for institution name
        :param soft_commit SoftCommit for Solr, default = True  True will refresh the view of the index in a more
        performant manner, without on-disk guarantees
        :return: None
        """

        start_time = datetime.now()

        try:
            pdf_fetch_start = datetime.now()
            pdf_doc = self._fetch_document(cl_id)
            pdf_fetch_elapsed = datetime.now() - pdf_fetch_start
            self.log.info('PDF Initialization elapsed time: [{0}]'.format(pdf_fetch_elapsed))

            pdf_parse_start = datetime.now()
            if pdf_doc.is_extractable:
                text = self._parse_pages(pdf_doc)
                self.log.info('PDF Parsing elapsed time: [{0}]'.format(datetime.now() - pdf_parse_start))

            else:
                raise PDFTextExtractionNotAllowed('File [{0}.pdf] is not extractable to a PDF document'.format(cl_id))

            if db_insert:
                self.log.info('Inserting to database')
                documents_indexed = self._save_to_db(text, cl_id, catalog_year, institution)
            else:
                self.log.info('Inserting to Solr')
                documents_indexed = self._solr(text, cl_id, catalog_year, institution, soft_commit)

            self.log.info('Total elapsed processing time: [{0}]'.format(datetime.now() - start_time))

            self._save_state(cl_id, 1, 1, documents_indexed)

            return {'state': states.SUCCESS, 'documents_indexed': documents_indexed}

        except (requests.RequestException, PDFException, ValueError, Exception) as e:
            self.log.error('{0} - {1}'.format(e, e.message))
            self._save_state(cl_id, 1, -1, index_message = e.message)
            raise e


class PdfPageIndexer(PdfElementIndexer):
    # Overrides the Solr insert and text extraction methods from PdfElementIndexer in which entire pages of content are inserted
    # into Solr, rather than indivitual LineText objects.

    def __init__(self):
        PdfElementIndexer.__init__(self)

    def parse_lt_objects(self, layout, index, text = []):

        self.log.debug('Processing LT objects for page [{0}]'.format(index))
        text_content = []
        page_text = {}  # k = (x0, x1) of bounding box, v = list of text strings within that column

        for obj in layout :
            if isinstance(obj, LTTextBox) or isinstance(obj, LTTextLine) :
                self.log.debug('[{0}] object found'.format(type(obj)))
                page_text = self._update_text_hash(page_text, obj)
            elif isinstance(obj, LTFigure) :
                # LTFigure objects are containers for other LT* objects, so recurse through children
                self.log.debug('LTFigure object found, recursing to process children nodes')
                text_content.append(self.parse_lt_objects(obj, index, text_content))

        self.log.debug('Page [{0}] extracted'.format(index))

        for k, v in sorted([(key, val) for (key, val) in page_text.items()]):
            text_content.append(''.join(v))

        return ' '.join(text_content)

    def _solr(self, data, cl_id, catalog_year, institution, soft_commit = True):
        self.log.info('Starting SOLR indexing with instance URL [{0}]'.format(SOLR))
        solr_instance = pysolr.Solr(PDF_PAGE_SOLR, timeout = 360)
        solr_data = []
        indexed_date = datetime.now().strftime('%c')

        for idx, val in enumerate(data):
            solr_data.append({
                'institution': institution,
                'catalog_year': catalog_year,
                'catalog_link': cl_id,
                'page': idx + 1,
                'indexed_date': indexed_date,
                'content': val
            })

        solr_instance.delete(q = 'catalog_link:{}'.format(cl_id))
        solr_instance.add(solr_data, waitSearcher = True)
        return len(solr_data)

    def _save_to_db(self, data, cl_id, catalog_year, institution):
        self.log.info('Starting saving data to database')

        document_list = []
        indexed_date = datetime.now().strftime('%c')
        page_number = 1
        for idx, val in enumerate(data):
            self.log.debug('Adding page [{}] to database'.format(page_number))
            entry = PdfIndexDocument(hash_id = '',
                                     bounds = '',
                                     page = page_number,
                                     content = val,
                                     catalog_link = cl_id,
                                     catalog_year = catalog_year,
                                     institution = institution,
                                     indexed_date = indexed_date)
            page_number += 1
            document_list.append(entry)

        PdfIndexDocument.objects.bulk_create(document_list)
        return len(document_list)


