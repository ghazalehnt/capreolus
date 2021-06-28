import os

import torch

from capreolus import Trainer, Searcher


class UnsupervisedTrainer(Trainer):
    name = "unsupervised"
    dependencies = []

    @staticmethod
    def config():
        pass

    def train(self, *args, **kwargs):
        return None

    def load_best_model(self, *args, **kwargs):
        return None

    def predict(self, reranker, pred_data, pred_fn):
        """Predict query-document scores on `pred_data` using `model` and write a corresponding run file to `pred_fn`

        Args:
           model (Reranker): a Reranker implementing query(query, docids)
           pred_data (IterableDataset): data to predict on
           pred_fn (Path): path to write the prediction run file to

        Returns:
           TREC Run

        """

        preds = {}
        pred_dataloader = torch.utils.data.DataLoader(pred_data, batch_size=None, pin_memory=False, num_workers=0)
        for idx, d in enumerate(pred_dataloader):
            qid, docid = d["qid"], d["posdocid"]
            scores = reranker.test(d)

            assert len(scores) == 1
            score = scores[0]

            preds.setdefault(qid, {})[docid] = score

        os.makedirs(os.path.dirname(pred_fn), exist_ok=True)
        Searcher.write_trec_run(preds, pred_fn)

        return preds
