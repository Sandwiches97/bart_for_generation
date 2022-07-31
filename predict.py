import pandas as pd
import re
from zhon.hanzi import punctuation
from transformers import BertTokenizer, BartForConditionalGeneration
import numpy as np
from rouge import Rouge
from snownlp import SnowNLP

def load_model(model_path:str):
    model_checkpoint = model_path
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    return tokenizer, model


def abstract_generate(sentence, max_words_number=150):
    print(sentence)
    sentence = re.sub("['“”''（）'‘\n’'\xa0']", '', sentence)
    sentence = re.sub("['，']", ',', sentence)
    count, abstract = 0, []
    for i in range(1, 15):
        text = SnowNLP(sentence).summary(i)
        count += len(text[i-1])
        if count >= max_words_number:
            abstract = SnowNLP(sentence).summary(i)
            keywords = SnowNLP(sentence).keywords(5)
            break
        else:
            abstract = ""
            keywords = ""
            abstract = '。'.join(abstract)
    abstract = re.sub("[',']", '，', abstract)
    return abstract, keywords

def generate_sample(sample:str, tokenizer, model, num_beam=3, mode="sample"):
    input = tokenizer(sample, max_length=512, truncation=True, return_tensors="pt")
    del input["token_type_ids"]
    if mode=="search":
        output = model.generate(**input, num_beams=num_beam, max_length=32, repetition_penalty=5.0, no_repeat_ngram_size=2)
    elif mode=="p": output = model.generate(**input, do_sample=True, top_p=0.9, max_length=32, repetition_penalty=5.0, no_repeat_ngram_size=2)
    elif mode=="k": output = model.generate(**input, do_sample=True, top_k=20, temperature=0.5, max_length=32, repetition_penalty=5.0, no_repeat_ngram_size=2)
    else: raise ValueError("请输入正确的mode：search, p, k")
    summary = tokenizer.decode(output[0]).split('[SEP]')[1].replace("[CLS]", '').replace(' ', '')
    return summary

def data_clean(text):
    text = re.sub("[{}]+".format(punctuation), '',text)
    text = re.sub("['\n', ' ', '\xa0']", '', text)
    return text

def save_result(path, tokenizer, model):
    """ 保存 dev.json 的预测结果

    :param path:
    :param tokenizer:
    :param model:
    :return:
    """
    data = pd.read_json(path)
    contents = data["content"]
    data["content"] = [re.sub("[{}]+".format(punctuation), '', content) for content in contents]
    data["content"] = [re.sub("['\n', ' ', '\xa0']", '', content) for content in contents]
    data["title"] = [re.sub("[{}]+".format(punctuation), '', title) for title in data["title"]]
    for i in range(len(data["title"])):
        data["content"][i] = generate_sample(data["content"][i], tokenizer, model)
    data.to_csv("predict_topP.csv")

def get_one_input(tokenizer, model):
    while(True):
        text = input("请输入一段文章 (退出请输入q)：")
        if text == 'q': break
        text = data_clean(text)
        title = generate_sample(text, tokenizer, model)
        print(f"\n文章标题：{title}\n", )

def get_txt_input(tokenizer, model):
    path = input("请输入txt存放地址：")
    f = open(path, encoding="utf-8")
    text = []
    for line in f:
        text.append(line.strip())
    text = str(text)
    text = data_clean(text)
    title = generate_sample(text, tokenizer, model)
    print(f"\n文章标题：{title}\n", )


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

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

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    # result["gen_len"] = np.mean(prediction_lens)

    result = {'rouge1':np.sum(mydic["rouge1"]["f"])/len(decoded_labels), 'rouge2':np.sum(mydic["rouge2"]["f"])/len(decoded_labels),
              'rougeL':np.sum(mydic["rougeL"]["f"])/len(decoded_labels), 'rougeLsum':0, 'gen_len':np.mean(prediction_lens)}

    return {k: round(v, 4) for k, v in result.items()}

def main():
    tokenizer, model = load_model("exp_dir/sum1/model/checkpoint-6700")

    while True:
        print("请进行模式选择，如果您需要在线对一段文章提取标题，输入1；如果您需要对txt文本生成标题，请输入2；退出请输入0：")
        mode_select = int(input("模式："))
        if mode_select == 1:
            get_one_input(tokenizer, model)
        elif mode_select == 2:
            get_txt_input(tokenizer, model)
        elif mode_select == 0:
            break
        else:
            print("请输入正确的选项（1或者2，退出请输入0）！")



if __name__ == "__main__":
    tokenizer, model = load_model("exp_dir/sum1/model/checkpoint-6700")
    path = "data/dev.json"
    save_result(path, tokenizer=tokenizer, model=model)
    # main()
    pass