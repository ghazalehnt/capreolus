import os
import pickle
from collections import Counter, defaultdict

import numpy as np

from capreolus import ConfigOption, Dependency
from capreolus.utils.loginit import get_logger

from . import Extractor
from ..utils.recbole_model import ItemLM

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Extractor.register
class DocStats(Extractor):
    """TODO"""

    module_name = "docstats"
    dependencies = [
        Dependency(key="benchmark", module="benchmark", name=None),
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(
            key="backgroundindex", module="index", name="anserinicorpus"
        ),
        Dependency(key="tokenizer", module="tokenizer", name="anserini", default_config_overrides={"keepstops": False}),
    ]
    config_spec = [
    ]
    pad = 0
    pad_tok = "<pad>"

    def exist(self):
        return hasattr(self, "doc_tf")

    def preprocess(self, qids, docids, topics, qdocs=None):
        if self.exist():
            return
        logger.debug("Opening background index")
        self.backgroundindex.open()

        logger.debug("tokenizing queries")
        self.qid_term_frequencies = {}
        self.qid_termprob = {}
        for qid in qids:
            qtext = topics[qid]
            qtext = qtext.replace("[", "")
            qtext = qtext.replace("]", "")
            qtext = qtext.strip()
            query = self.tokenizer.tokenize(qtext)
            q_count = Counter(query)
            self.qid_term_frequencies[qid] = {k: v for k, v in q_count.items()}
            self.qid_termprob[qid] = {k: (v / len(query)) for k, v in q_count.items()}

        logger.debug("loading documents")
        self.doc_tf = {}
        self.doc_len = {}
        if self.benchmark.collection.module_name == "kitt":
            for docid in docids:
                doc = self.tokenizer.tokenize(self.index.get_doc(docid)) # TODO change
                self.doc_tf[docid] = Counter(doc)
                self.doc_len[docid] = len(doc)
        elif self.benchmark.collection.module_name == "kitt_inferred":
            book_KITT_urls = {}
            with open("/GW/PSR/work/data_personalization/documents/book/book_url_ids.txt", 'r') as f:
                for l in f.readlines():
                    sp = l.split(",")
                    book_KITT_urls["book_" + sp[0].strip()] = sp[1].strip()
            recbole_path = "/GW/PSR/work/ghazaleh/RecBole/"
            rec_model = ItemLM(f"{recbole_path}saved/JOINTSRMFSPARSE-Jun-25-2021_11-40-14.pth", [f"{recbole_path}config_RO_RS_JSR.yml"],
                           "JOINTSRMFSPARSE", "KITT_goodreads_rated", k=100, step=200000, load_docs=book_KITT_urls.values())
            for docid in docids:
                b_url = book_KITT_urls[docid]
                lm, title, url, recbole_id = rec_model.get_terms_url(b_url)
                if lm is None:
                    print(b_url)
                    continue
                min_p = min(lm, key=lambda x: x[1])[1]
                estimated_length = 1/min_p
                self.doc_tf[docid] = {i[0]: round(i[1] * estimated_length) for i in lm}
                self.doc_len[docid] = sum(self.doc_tf[docid].values())

        logger.debug("calculating average document length")
        self.query_avg_doc_len = {}
        for qid, docs in qdocs.items():
            doclen = 0
            for docid in docs:
                doclen += self.doc_len[docid]
            self.query_avg_doc_len[qid] = doclen / len(docs)

        logger.debug("extractor DONE")

    def background_idf(self, term):# TODO could be replaced by that: the index itself has a function for idf, but it has a +1...
        df = self.backgroundindex.get_df(term)
        total_docs = self.backgroundindex.numdocs
        return np.log10((total_docs - df + 0.5) / (df + 0.5))