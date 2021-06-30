from os.path import join
import torch
from recbole.config import Config
from recbole.utils import get_model, init_seed
import gensim
import gensim.downloader as api
from recbole.data import create_dataset, data_preparation
URL_FIELD = "item_url"


class ItemLM:
    def __init__(self, checkpoint_file, config_files, model_name, dataset_name, k=20, step=5000, load_docs=None):
        self.k = k
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        config = Config(model=model_name, dataset=dataset_name, config_file_list=config_files, config_dict=None)
        init_seed(config['seed'], config['reproducibility'])
        self.dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, self.dataset)
        model = get_model(config['model'])(config, train_data).to(config['device'])
        model.load_state_dict(checkpoint['state_dict'])

        item_ids = self.dataset.get_item_feature()['item_id']
        items = model.item_embedding(item_ids)
        item_identifiers = self.dataset.get_item_feature()[URL_FIELD]
        self.item_identifiers = self.dataset.id2token(URL_FIELD, item_identifiers)
        self.url_id = {}
        for i in range(1, len(self.item_identifiers)):
            self.url_id[self.item_identifiers[i]] = i
        self.item_names = self.dataset.get_item_feature()['item_title']

        if load_docs is not None:
            item_ids = list(set(self.url_id[url] for url in load_docs))

        self.items_top_terms = {}
        print("making item lm...")
        s = 1
        e = step
        while e <= len(item_ids):
            print(f"{s}:{e}")
            batch_ids = item_ids[s:e]
            batch_items = items[batch_ids].detach().clone()
            batch_lms = torch.matmul(batch_items, model.word_embedding.weight.T)
            batch_lms = torch.softmax(batch_lms, 1)
            batch_lms = batch_lms.topk(k, dim=1)
            probs_normalized_topk = (batch_lms.values.T / batch_lms.values.sum(1)).T
            for i in range(len(batch_ids)):
                self.items_top_terms[int(batch_ids[i])] = {"indices": batch_lms.indices[i], "probs": probs_normalized_topk[i]}
            s = e
            e += step
            if e > len(item_ids) > s:
                e = len(item_ids)
        print("done")
        pretrained_embedding_name = "glove-wiki-gigaword-50"
        model_path = api.load(pretrained_embedding_name, return_path=True)
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(model_path)

    def get_terms_url(self, item_url):
        if item_url not in self.url_id:
            print(f"We don't have this url in our model {item_url}, because it did not have any reviews it was not crawled. TODO do that... what if it doesn't have description as well?")
            return None, None, None, None
        item_id = self.url_id[item_url]
        top_terms = self.items_top_terms[item_id]
        name_tokens = self.item_names[item_id]
        item_name = self.dataset.id2token("item_title", name_tokens)
        ret = []
        for j in range(self.k):
            idx = top_terms["indices"][j]
            ret.append((self.w2v_model.index_to_key[idx], float(top_terms["probs"][j])))
        return ret, item_name, item_url, item_id

    def get_terms_recbole_id(self, item_id):
        top_terms = self.items_top_terms[item_id]
        item_url = self.item_identifiers[item_id]
        name_tokens = self.item_names[item_id]
        item_name = self.dataset.id2token("item_title", name_tokens)
        ret = []
        for j in range(self.k):
            idx = top_terms["indices"][j]
            ret.append((self.w2v_model.index_to_key[idx], float(top_terms["probs"][j])))
        return ret, item_name, item_url, item_id
