import pandas as pd
from  TextRank4ZH.textrank4zh import TextRank4Sentence as MyTextRank
from textrank4zh import TextRank4Sentence
import re

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
    # tr4s = MyTextRank()
    tr4s = TextRank4Sentence()
    for i in range(len(data)):
        data.loc[i, "content"] = re.sub("['\n', ' ', '\xa0', '\u3000', '■']", '', data.loc[i, "content"] )
        data.loc[i, "content"] = re.sub("……|…", '。', data.loc[i, "content"])
        data.loc[i, "content"] = re.sub("——|—", '，', data.loc[i, "content"])
        data.loc[i, "content"] = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|\\(.*?\\)", '', data.loc[i, "content"])
        if data['len'].values[i] < 520: continue
        text = data["content"].values[i]#[:int(mean_len+var_len//0.5)]

        tr4s.analyze(text=text,
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

if __name__ == "__main__":
    main()