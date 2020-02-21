import random
import torch.utils.data

from capreolus.utils.exceptions import MissingDocError
from capreolus.utils.loginit import get_logger


logger = get_logger(__name__)


class TrainDataset(torch.utils.data.IterableDataset):
    """
    Samples training data. Intended to be used with a pytorch DataLoader
    """

    def __init__(self, training_judgments, extractor):
        self.extractor = extractor
        self.iterations = 0

        self.qid_to_reldocs = {
            qid: [docid for docid, label in doclabels.items() if label > 0] for qid, doclabels in training_judgments.items()
        }

        self.qid_to_negdocs = {
            qid: [docid for docid, label in doclabels.items() if label <= 0] for qid, doclabels in training_judgments.items()
        }

        # remove any qids that do not have both relevant and non-relevant documents for training
        for qid in list(training_judgments.keys()):
            posdocs = len(self.qid_to_reldocs[qid])
            negdocs = len(self.qid_to_negdocs[qid])

            if posdocs == 0 or negdocs == 0:
                logger.warning("removing training qid=%s with %s positive docs and %s negative docs", qid, posdocs, negdocs)
                del self.qid_to_reldocs[qid]
                del self.qid_to_negdocs[qid]

    def __iter__(self):
        """
        Returns: Triplets of the form (query_feature, posdoc_feature, negdoc_feature)
        """

        # Convert each query and doc id to the corresponding feature/embedding and yield
        while True:
            all_qids = sorted(self.qid_to_reldocs)
            random.seed(self.iterations)
            random.shuffle(all_qids)

            for qid in all_qids:
                posdocid = random.choice(self.qid_to_reldocs[qid])
                negdocid = random.choice(self.qid_to_negdocs[qid])

                try:
                    query_feature, posdoc_feature, negdoc_feature = self.extractor.id2vec(qid, posdocid, negdocid)
                    yield {"query": query_feature, "posdoc": posdoc_feature, "negdoc": negdoc_feature}
                except MissingDocError:
                    # at training time we warn but ignore on missing docs
                    logger.warning(
                        "skipping training pair with missing features: qid=%s posid=%s negid=%s", qid, posdocid, negdocid
                    )


class PredDataset(torch.utils.data.IterableDataset):
    """
    Creates a Dataset for evaluation (test) data to be used with a pytorch DataLoader
    """

    def __init__(self, pred_pairs, extractor):
        def genf():
            for qid, docids in pred_pairs.items():
                for docid in docids:
                    try:
                        query_feature, posdoc_feature = extractor.id2vec(qid, docid)
                        yield {"query": query_feature, "posdoc": posdoc_feature}
                    except MissingDocError:
                        # when predictiong we raise an exception on missing docs, as this may invalidate results
                        logger.error("got none features for prediction: qid=%s posid=%s", qid, docid)
                        raise

        self.generator_func = genf

    def __iter__(self):
        """
        Returns: Tuples of the form (query_feature, posdoc_feature)
        """

        return self.generator_func()
