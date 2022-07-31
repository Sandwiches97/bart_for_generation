import re
import string
from datasets import Dataset
import jieba
import pandas as pd
from zhon.hanzi import punctuation
from transformers import BertTokenizer, BartForConditionalGeneration
import numpy as np
from rouge import Rouge

def dataLoad(name: str):
    path = "data/" + name + '.json'
    data = pd.read_json(path)
    contents = data["content"]
    data["content"] = [re.sub("[{}]+".format(punctuation), '', content) for content in contents]
    data["content"] = [re.sub("['\n', ' ', '\xa0']", '', content) for content in contents]
    data["title"] = [re.sub("[{}]+".format(punctuation), '', title) for title in data["title"] ]

    # contents[:] = [jieba.lcut(content) for content in contents]
    # titles[:] = [jieba.lcut(title) for title in titles]
    return data

def data_clean(text):
    text = re.sub("[{}]+".format(punctuation), '',text)
    text = re.sub("['\n', ' ', '\xa0']", '', text)
    return text

def load_model(model_path:str):
    model_checkpoint = model_path
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    return tokenizer, model

def generate_sample(sample:str, tokenizer, model, num_beam=3):
    input = tokenizer(sample, max_length=512, truncation=True, return_tensors="pt")
    del input["token_type_ids"]
    output = model.generate(**input, num_beams=num_beam, max_length=32, repetition_penalty=5.0)
    summary = tokenizer.decode(output[0]).split('[SEP]')[1].replace("[CLS]", '').replace(' ', '')
    return summary

def compute_metrics(eval_pred, tokenizer):
    decoded_preds, decoded_labels = eval_pred


    r = Rouge()
    mydic = {'rouge1': {'f':[], 'p':[], 'r':[]}, 'rouge2': {'f':[], 'p':[], 'r':[]}, 'rougeL': {'f':[], 'p':[], 'r':[]}}
    for i in range(len(decoded_labels)):
        if (decoded_preds[i]==''): continue
        mylist = r.get_scores(decoded_preds[i], decoded_labels[i])
        r1 = mylist[0]['rouge-1']
        r2 = mylist[0]['rouge-2']
        rL = mylist[0]['rouge-l']
        for s in ['f', 'p', 'r']:
            mydic['rouge1'][s].append(r1[s])
            mydic['rouge2'][s].append(r2[s])
            mydic['rougeL'][s].append(rL[s])

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in decoded_preds]
    # result["gen_len"] = np.mean(prediction_lens)

    result = {'rouge1':np.sum(mydic["rouge1"]["f"])/len(decoded_labels), 'rouge2':np.sum(mydic["rouge2"]["f"])/len(decoded_labels),
              'rougeL':np.sum(mydic["rougeL"]["f"])/len(decoded_labels), 'rougeLsum':0, 'gen_len':np.mean(prediction_lens)}

    return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
    data = pd.read_csv("predict2.csv", encoding="utf-8", usecols=[1, 2, 3])
    data["title_len"] = data["title"].apply(lambda x: len(x))
    data = data.sort_values(by='title_len', ascending=False)
    pass
