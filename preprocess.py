import re
import pandas as pd
import numpy as np
import string
import torch
from zhon.hanzi import punctuation
from datasets import Dataset
import jieba
import tqdm
from snownlp import SnowNLP
from textrank4zh import TextRank4Sentence
from functools import reduce
from TextRank4ZH.textrank4zh import TextRank4Sentence as MyTextRank
import matplotlib.pyplot as plt


def remove_punc(text):
    panc = re.compile(r'，')
    return panc.sub(r'', text)

def data_process(name: str):
    path = "data/" + name + '.json'
    data = pd.read_json(path)

    for i, text in tqdm.tqdm(enumerate(data["content"])):
        lst = divedeBlock(text, BLOCK_SIZE=255)
        contents = ""
        text = re.sub("，", ",", text)
        for j in range(len(lst) - 1, -1, -1):
            content = SnowNLP(text[lst[j][0]:lst[j][1]])
            tmp = content.summary(1)
            contents += tmp[0] + "。"
        contents = re.sub(",", "，", contents)
        contents = re.sub("”", '"', contents)
        contents = re.sub("“", '"', contents)
        contents = re.sub("，。", "。", contents)
        contents = re.sub("\n|\xa0|\u3000", "", contents)
        contents = re.sub("（.*）", "", contents)
        data["content"][i] = contents
        # print(contents)
    return data

def show_dist(path:str, mode="json"):
    import numpy as np
    import seaborn as sns

    if mode=="json":
        data = pd.read_json(".\\data\\" + path + ".json")
    else:
        data = pd.read_csv(".\\data\\" + path + ".csv", usecols=[1, 2])
    """ rouge 1 """
    mylist = []
    for i in range(len(data)):
        lst_one = [0] * len(data.loc[i, 'content'])
        for j in range(len(lst_one)):
            if data.loc[i, "content"][j] in data.loc[i, "title"]:
                lst_one[j] += 1
        mylist.append(lst_one)

    # 长条图
    mylistSum = list(map(lambda x: sum(x[:512]), mylist))
    print(np.mean(mylistSum))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(mylistSum)), mylistSum)

    ## 热力图
    # npdata = np.zeros((len(mylist), 2000))
    # for i in range(len(mylist)):
    #     y = 2000
    #     if len(mylist[i]) < y: y = len(mylist[i])
    #     for j in range(y):
    #         npdata[i][j] = mylist[i][j]
    # ax1 = sns.heatmap(npdata)


    """ rouge 2"""

    data["len"] = data["content"].apply(lambda x: len(x))
    # data["title"] = data["title"].apply(lambda x: list(jieba.cut(x)))
    # data["content"] = data["content"].apply(lambda x: list(jieba.cut(x)))
    # mylist = []
    # for i in range(len(data)):
    #     setTMP = set(data.loc[i, "title"])
    #     lst = [0]*len(data.loc[i, 'content'])
    #     for j in range(len(lst)):
    #         if data.loc[i, "content"][j] in setTMP:
    #             lst[j] += 1
    #     # preSum = [0]*(len(lst)+1)
    #     # for j in range(1, len(preSum)):
    #     #     preSum[j] = preSum[j-1] + lst[j-1]
    #     mylist.append(lst)
    # npdata = np.zeros((len(mylist), 2000))
    # for i in range(len(mylist)):
    #     y = 2000
    #     if len(mylist[i]) < y: y = len(mylist[i])
    #     for j in range(y):
    #         npdata[i][j] = mylist[i][j]
    # ax2 = sns.heatmap(npdata)


    plt.subplot(1, 2, 2)
    plt.bar(range(len(data)), data['len'].values)

    plt.show()
    pass

def dataLoad(name: str, dtype: str, mode=None):
    path = "data/" + name + dtype
    data = pd.read_json(path)
    contents = data["content"]
    # data["content"] = [re.sub("[{}]+".format(punctuation), '', content) for content in contents]
    data["content"] = [re.sub("…", '。', content) for content in contents]
    data["content"] = [re.sub("['\n', ' ', '\xa0', '\u3000', '—']", '', content) for content in contents]
    data["content"] = [re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|\\(.*?\\)", '', content) for content in contents]
    # data["title"] = [re.sub("[{}]+".format(punctuation), '', title) for title in data["title"] ]

    if mode == "save_slipe":
        size = data["title"].size
        dic = pd.DataFrame(columns=["title", "content"])
        for i in range(size):
            content = data["content"][i]
            title = data["title"][i]
            while len(content) >512:
                tmp = content[:512]
                content = content[256:]
                dic = dic.append({"title": title, "content": tmp}, ignore_index=True)
            dic = dic.append({"title": title, "content": content}, ignore_index=True)
        return dic
    else: return data

class Data:
    def __init__(self, tokenizer):
        self.max_input_length = 512
        self.max_target_length = 32
        self.tokenizer = tokenizer

    def preProcess(self):
        train_dic = pd.read_csv('data/train_17brkSoft_all.csv', usecols=[1, 2])
        test_dic = pd.read_csv('data/dev_17brkSoft_all.csv', usecols=[1, 2])

        def preprocess_function(example):
            inputs = example['content']
            model_inputs = self.tokenizer(inputs, max_length=self.max_input_length,
                                          padding='max_length', truncation=True)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(example['title'], max_length=self.max_target_length,
                                        padding='max_length', truncation=True)
            model_inputs['labels'] = labels['input_ids']
            return model_inputs

        train_dataset = Dataset.from_dict(train_dic)
        test_dataset = Dataset.from_dict(test_dic)
        tokenized_train_dataset = train_dataset.map(preprocess_function)
        tokenized_test_dataset = test_dataset.map(preprocess_function)
        return tokenized_train_dataset, tokenized_test_dataset

def divedeBlock(d, BLOCK_SIZE):
    end_tokens = {'\n': 0, '。': 1, '？': 1, '！': 1, '，': 2}
    # for k, v in list(end_tokens.items()):
    #     end_tokens['Ġ' + k] = v
    sen_cost, break_cost = 4, 8
    poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if tok in end_tokens]
    poses.insert(0, (-1, 0))  # 在起始位置，插入（-1，0）
    if poses[-1][0] < len(d) - 1:  # 最后一个分隔符的索引，如果不是最后一个字符
        poses.append((len(d) - 1, 0))  # 则，把最后一个字符索引加进去
    x = 0
    while x < len(poses) - 1:
        if poses[x + 1][0] - poses[x][0] > BLOCK_SIZE:
            poses.insert(x + 1, (poses[x][0] + BLOCK_SIZE, break_cost))
        x += 1
    # simple dynamic programming
    best = [(0, 0)]
    for i, (p, cost) in enumerate(poses):
        if i == 0:
            continue  # 跳过起始项 （-1，0）
        best.append((-1, 100000))
        for j in range(i - 1, -1, -1):
            if p - poses[j][0] > BLOCK_SIZE:
                break
            value = best[j][1] + cost + sen_cost
            if value < best[i][1]:
                best[i] = (j, value)
        assert best[i][0] >= 0
    intervals, x = [], len(poses) - 1
    while x > 0:  # x: 右边界
        l = poses[best[x][0]][0]
        intervals.append((l + 1, poses[x][0] + 1))
        x = best[x][0]
    return intervals

def divedeTitle():
    pass

def similar(data: str, label: str):
    ans = 0
    for i in range(len(data)):
        if data[i] in label:
            ans += 1
    return ans


def textrank_sentences(path:str):
    data = pd.read_json("data/" + path + ".json")
    data["len"] = data["content"].apply(lambda x: len(x))

    mean_len = data["len"].mean()
    var_len = data["len"].std()
    tr4s = MyTextRank()
    # tr4s = TextRank4Sentence()
    for i in range(len(data)):
        data.loc[i, "content"] = re.sub("['\n', ' ', '\xa0', '\u3000', '■']", '', data.loc[i, "content"] )
        data.loc[i, "content"] = re.sub("……|…", '。', data.loc[i, "content"])
        data.loc[i, "content"] = re.sub("——|—", '，', data.loc[i, "content"])
        data.loc[i, "content"] = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|\\(.*?\\)", '', data.loc[i, "content"])
        if data['len'].values[i] < 520: continue
        text = data["content"].values[i]#[:int(mean_len+var_len//0.5)]

        tr4s.analyze(text=text,
                     block_size=12,
                     lower=True, source='all_filters')
        sentences = tr4s.get_key_sentences(num=70)
        sentences_len = list(map(lambda x: len(x["sentence"]), sentences))
        accumulate = [0]*(len(sentences_len)+1)
        idx = len(sentences_len)
        for j in range(len(sentences_len)):
            accumulate[j+1] = accumulate[j] + sentences_len[j]
            if accumulate[j+1] > 510:
                idx = j+1
                break
        new_sent = sorted(sentences[:idx], key=lambda x: x["index"])
        print(data.loc[i, "title"])
        print(data.loc[i, "content"][:512])
        data.loc[i, "content"] = "".join(list(map(lambda x: x["sentence"] #+"。"
                                                  , new_sent)))
        print(data.loc[i, "content"])
        print(similar(data.loc[i, "content"], data.loc[i, "title"]))
    data.to_csv("data/" + path + "_8brkSoft_all" + ".csv", encoding="utf-8")

    # for item in tr4s.get_key_sentences(num=10):
    #     print(item.index, item.weight, item.sentence)

def main():
    # show_dist("dev_8brk_all", mode="csv")
    textrank_sentences("dev")

    # data.to_csv('data/dev_block.csv', encoding="utf_8")
    # data = pd.read_csv("data/dev_trank.csv", usecols=[1,2])
    # data.to_csv("data/dev" + "_trank" + ".csv", columns=["title", "content"], encoding="utf-8")
    # data.to_json("data_test.json", force_ascii=False)
    # data["content"] = data["content"].apply(lambda x: jieba.lcut(x))

if __name__ == "__main__":
    main()

    # data = pd.read_csv("data/train_block.csv", usecols=[1, 2])
    # print(data)

    # t = "中国持续发力削减全球“赤字”"
    # BLOCK_SIZE = 255
    # lst = divedeBlock(s, BLOCK_SIZE)
    # contents = ""
    # s = re.sub("，", ",", s)
    # for j in range(len(lst)-1, -1, -1):
    #     content = SnowNLP(s[lst[j][0]:lst[j][1]])
    #     tmp = content.summary(1)
    #     contents +=  tmp[0] + "。"
    # print(len(contents))
    # contents = re.sub(",", "，", contents)
    # contents = re.sub("，。", "。", contents)
    # print(contents)