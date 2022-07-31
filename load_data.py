import re
import pandas as pd
import numpy as np
from zhon.hanzi import punctuation
from datasets import Dataset

def dataLoad(name: str):
    path =  'data/' + name + '.json'
    data = pd.read_json(path)
    contents, titles = data['content'], data['title']
    contents = np.array(contents.values)
    titles = np.array(titles.values)

    contents[:] = [re.sub("[{}]+".format(punctuation), '', content) for content in contents]
    contents[:] = [re.sub("['\n', ' ', '\xa0']", '', content) for content in contents]
    titles[:] = [re.sub("[{}]+".format(punctuation), '', title) for title in titles]
    # titles[:] = [re.sub("['\n', ' ', '\xa0']", '', title) for title in titles]

    raw_data = {'id': [], 'document': [], 'title': []}

    for i, content in enumerate(contents):
        raw_data['id'].append(i)
        raw_data['title'].append(titles[i])
        raw_data['document'].append(content)

    # contents[:] = [jieba.lcut(content) for content in contents]
    # titles[:] = [jieba.lcut(title) for title in titles]
    return raw_data


class Data:
    def __init__(self, tokenizer):
        self.max_input_length = 1024
        self.max_target_length = 30
        self.tokenizer = tokenizer

    def preProcess(self):
        train_dic = dataLoad('train')
        test_dic = dataLoad('dev')

        def preprocess_function(example):
            inputs = example['document']
            model_inputs = self.tokenizer(inputs, max_length=self.max_input_length,
                                          padding='max_length', trunctation=True)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(example['title'], max_length=self.max_target_length,
                                        padding='max_length', trunctation=True)
            model_inputs[labels] = labels['input_ids']
            return model_inputs

        train_dataset = Dataset.from_dict(train_dic)
        test_dataset = Dataset.from_dict(test_dic)
        tokenized_train_dataset = train_dataset.map(preprocess_function)
        tokenized_test_dataset = test_dataset.map(preprocess_function)
        return tokenized_train_dataset, tokenized_test_dataset


data = dataLoad("train")
print(len(data['document']))
print(data.keys())