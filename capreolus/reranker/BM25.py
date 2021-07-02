import json
import os
from os.path import join, exists

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Reranker.register
class BM25Reranker(Reranker):
    """ BM25 implemented in Python as a reranker. Tested only with PES20 benchmark.
        This mainly serves as a demonstration of how non-neural methods can be prototyped as rerankers.
    """

    module_name = "BM25"
    dependencies = [
        Dependency(key="extractor", module="extractor", name="docstats"),
        Dependency(key="trainer", module="trainer", name="unsupervised"),
    ]

    config_spec = [
        ConfigOption("b", 0.4, "b"),
        ConfigOption("k1", 0.9, "k1"),
        ConfigOption("c", None, "c"),
    ]

    def build_model(self):
        if not exists(self.get_cache_path()):
            os.makedirs(self.get_cache_path(), exist_ok=True)
        return self

    def get_docscore_cache_path(self):
        return self.get_cache_path() / 'docscores'

    def test(self, d):
        if not hasattr(self.extractor, "doc_len"):
            raise RuntimeError(
                "reranker's extractor has not been created yet. try running the task's train() method first.")

        # os.makedirs(self.get_docscore_cache_path(), exist_ok=True)

        # query = self["extractor"].qid2toks[d["qid"]]
        # avg_doc_len = self["extractor"].query_avg_doc_len[d["qid"]]
        # return [self.score_document(query, d["qid"], docid, avg_doc_len) for docid in [d["posdocid"]]] not used any more since it is more optimal to use tf
        querytf = self.extractor.qid_termprob[d["qid"]]
        avg_doc_len = self.extractor.query_avg_doc_len[d["qid"]]
        return [self.score_document_tf(querytf, d["qid"], docid, avg_doc_len) for docid in [d["posdocid"]]]

    def score_document_tf(self, querytf, qid, docid, avg_doc_len):
        uid = qid.split("_")[-1]
        term_scores = {}
        scoresum = 0
        for term, tf in querytf.items():
            termscore = self.score_document_term(term, docid, avg_doc_len, tf)
            # if termscore != 0 and self.extractor.domain_vocab_specific is not None:
            #     termscore *= self.extractor.domain_term_weight[term]
            # if termscore != 0 and self.extractor.query_vocab_specific is not None:
            #     if self.extractor.profile_term_weight_by == 'topic':
            #         termscore *= self.extractor.profile_term_weight[term]
            #     elif self.extractor.profile_term_weight_by == 'user':
            #         termscore *= self.extractor.profile_term_weight[uid][term]

            term_scores[term] = termscore
            scoresum += termscore

        outf = join(self.get_docscore_cache_path(), f"{qid}_{docid}")
        with open(outf, 'w') as f:
            term_scores["OVERALL_SCORE"] = scoresum
            sorted_scores = {k: v for k, v in sorted(term_scores.items(), key=lambda item: item[1], reverse=True)}
            # final_scores = {}
            # for k, v in sorted_scores.items():
                # domain_term_weight = "-"
                # if self.extractor.domain_vocab_specific is not None:
                #     domain_term_weight = self.extractor.domain_term_weight[k] if k in self[
                #         "extractor"].domain_term_weight else "-"
                # prof_term_weight = "-"
                # if self.extractor.query_vocab_specific is not None:
                #     if self.extractor.profile_term_weight_by == 'topic':
                #         prof_term_weight = self.extractor.profile_term_weight[k] if k in self[
                #             "extractor"].profile_term_weight else "-"
                #     elif self.extractor.profile_term_weight_by == 'user':
                #         prof_term_weight = self.extractor.profile_term_weight[uid][k] if k in self[
                #             "extractor"].profile_term_weight[uid] else "-"
                # final_scores[k] = (v, domain_term_weight, prof_term_weight)
            f.write(json.dumps(sorted_scores, indent=4))

        return scoresum

    def score_document_term(self, term, docid, avg_doc_len, query_tf):
        tf = self.extractor.doc_tf[docid].get(term, 0)
        numerator = tf * (self.config["k1"] + 1)
        denominator = tf + self.config["k1"] * (
                    1 - self.config["b"] + self.config["b"] * (self.extractor.doc_len[docid] / avg_doc_len))
        doctf = numerator / denominator

        if self.config["c"] is None:
            qtf = query_tf
        else:
            qtf = (query_tf * (self.config["c"] + 1)) / (query_tf + self.config["c"])

        idf = self.extractor.background_idf(term)

        return idf * doctf * qtf
