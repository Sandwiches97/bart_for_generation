import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import os
import pandas as pd
import re

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def data_process(name: str):
    path = "data/" + name + '.json'
    data = pd.read_json(path)
    contents = data["content"]
    # data["content"] = data["content"].apply(lambda text: remove_punc(text))
    # data["content"] = [re.sub("[{}]+".format(punctuation), '', content) for content in contents]
    data["content"] = [re.sub("['\n', '\xa0', '\u3000']", '', content) for content in contents]
    data["content"] = [re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|\\(.*?\\)", '', content) for content in contents]
    # data["content"] = [re.sub("”", '"', content) for content in contents]
    # data["content"] = [re.sub("“", '"', content) for content in contents]
    return data

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class TrainData(Dataset):
    def __init__(self, data, tokenizer, MAX_GEN, MAX_LEN, use_token=False):
        super(TrainData, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.MAX_GEN = MAX_GEN
        self.MAX_LEN = MAX_LEN
        self.use_token = use_token
        if self.use_token:
            self.tokenize_data(self.data)

    def tokenize_data(self, data):
        encode_title = self.tokenizer.batch_encode_plus(
            data["title"].values, max_length=self.MAX_GEN, padding="max_length", truncation=True,
            return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
        )
        encode_content = self.tokenizer.batch_encode_plus(
            data["content"].values, max_length=self.MAX_LEN, padding="max_length", truncation=True,
            return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
        )
        print("_______________the load for train-val tokens is  finished !________________")
        self.title_idx = encode_title["input_ids"]
        self.title_mask = encode_title["attention_mask"]
        self.content_idx = encode_content["input_ids"]
        self.content_mask = encode_content["attention_mask"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return {"title_idx": self.title_idx[item],
                "title_mask": self.title_mask[item],
                "content_idx": self.content_idx[item],
                "content_mask": self.content_mask[item]} if self.use_token else \
            (self.data["content"].values[item], self.data["title"].values[item])

class TestData(Dataset):
    def __init__(self, data, tokenizer, MAX_LEN=512, use_token=False):
        super(TestData, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.use_token = use_token
        self.MAX_LEN = MAX_LEN
        if self.use_token:
            self.tokenize_data(self.data)

    def tokenize_data(self, data):
        encode_content = self.tokenizer.batch_encode_plus(
            data["content"].values, max_length=self.MAX_LEN, padding="max_length", truncation=True,
            return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
        )
        print("_______________the load for test tokens is  finished !________________")
        self.content_idx = encode_content["input_ids"]
        self.content_mask = encode_content["attention_mask"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return {"content_idx": self.content_idx[item],
                "content_mask": self.content_mask[item]} if self.use_token else \
            self.data["content"].values[item]

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weights_loss = (unweighted_loss * weights).mean(dim=1)
        return weights_loss




class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self): self.tlk = time.time()
    def stop(self):
        self.times.append(time.time()-self.tlk)
        return self.times[-1]
    def avg(self): return sum(self.times)/len(self.times)
    def cumsum(self): return np.array(self.times).cumsum().tolist()

class Accumulator:
    def __init__(self, n):
        self.var = [0.0]*n

    def add(self, *args):
        self.var = [a + float(b) for a, b in zip(self.var, args)]

    def reset(self):
        self.var = [0.0]*len(self.var)

    def __getitem__(self, item):
        return self.var[item]

def grad_clipping(net, theta):
    """Clip the gradient.

    Defined in :numref:`sec_rnn_scratch`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm