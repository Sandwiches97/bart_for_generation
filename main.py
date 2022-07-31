import time
import math
import torch
from torch import nn
from torch.nn import modules
from torch.utils.data import Dataset, DataLoader
import os
from transformers import BertTokenizer, BartForConditionalGeneration
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from preprocess import Data
from load_data import dataLoad
import pandas as pd
import re
from transformers import get_linear_schedule_with_warmup
from functools import reduce
from zhon.hanzi import punctuation
from metric import compute
import jieba
from sklearn.model_selection import KFold
import numpy as np
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
from rouge import Rouge
from My_utilits import Timer, Accumulator, grad_clipping, TrainData, TestData, try_gpu, data_process, create_dir_not_exist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_pt_utils import LabelSmoother
from typing import Tuple

def freeze(module):
    """ freezes module's parameters. """
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_freezed_parameters(module):
    """
    Returns names of freezed parameters of the given module.
    """

    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)

    return freezed_parameters

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
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

    prediction_lens = [torch.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    result = {'rouge1':sum(mydic["rouge1"]["f"])/len(decoded_labels), 'rouge2':sum(mydic["rouge2"]["f"])/len(decoded_labels),
              'rougeL':sum(mydic["rougeL"]["f"])/len(decoded_labels), 'rougeLsum':0, 'gen_len':sum(prediction_lens)/len(prediction_lens)}

    return {k: round(float(v), 4) for k, v in result.items()}

def compute_metrics_np(eval_pred):
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

    result = {'rouge1':sum(mydic["rouge1"]["f"])/len(decoded_labels), 'rouge2':sum(mydic["rouge2"]["f"])/len(decoded_labels),
              'rougeL':sum(mydic["rougeL"]["f"])/len(decoded_labels), 'rougeLsum':0, 'gen_len':sum(prediction_lens)/len(prediction_lens)}
    return {k: round(float(v), 4) for k, v in result.items()}

def get_optimizer_scheduler(trainer, len_dataloader):
    num_update_steps_per_epoch = len_dataloader // myargs.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    max_steps = math.ceil(EPOCHES * num_update_steps_per_epoch)
    trainer.create_optimizer_and_scheduler(max_steps) # 7310
    optimizer = trainer.optimizer
    scheduler = trainer.lr_scheduler
    return optimizer, scheduler

class MyBart(nn.Module):
    def __init__(self, name):
        super(MyBart, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(name)

    def forward(self,
                input_ids,
                attention_mask,
                decode_input_ids,
                # decoder_attention_mask,
                label=None):
        output = self.bart(input_ids, attention_mask, decode_input_ids,
                           # decoder_attention_mask,
                           labels=label)
        return output

def train_batch(net, train_dl, optimizer, tokenizer, scheduler, metric, use_token:bool, writer, epoch:int):
    label_smoother = LabelSmoother(epsilon=LABEL_SMOOTHING)
    gradient_accumulation_steps = myargs.gradient_accumulation_steps
    steps = len(train_dl)
    for n, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
        optimizer.zero_grad()
        if use_token:
            input_ids = batch["content_idx"].cuda()
            attention_mask = batch["content_mask"].cuda()
            Y_ids = batch["title_idx"][:, :-1].contiguous().cuda()
            lm_labels = batch["title_idx"][:, 1:].clone().detach().cuda()
            lm_labels[batch["title_idx"][:, 1:] == tokenizer.pad_token_id] = -100
            Y_attention_mask = batch["title_mask"].cuda()
        else:
            text = batch[0]
            label = batch[1]
            encoded = tokenizer.batch_encode_plus(
                list(text), max_length=MAX_LEN, padding="max_length", truncation=True,
                return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
            )
            encoded_Y = tokenizer.batch_encode_plus(
                list(label), max_length=MAX_GEN, padding="max_length", truncation=True,
                return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
            )
            input_ids = encoded["input_ids"].cuda()
            attention_mask = encoded["attention_mask"].cuda()
            Y_ids = encoded_Y["input_ids"][:, :-1].contiguous().cuda()
            lm_labels = encoded_Y["input_ids"][:, 1:].clone().detach().cuda()
            lm_labels[encoded_Y["input_ids"][:, 1:] == tokenizer.pad_token_id] = -100
            Y_attention_mask = encoded_Y["attention_mask"].cuda()

        preds = net(input_ids=input_ids,
                    attention_mask=attention_mask,
                    decode_input_ids=Y_ids, # 102, 101, ...
                    label=lm_labels # 101, ....
                    )
        # l = criterion(preds.logits, Y_ids, Y_attention_mask.sum(dim=1))
        # print('\n', tokenizer.batch_decode(preds.logits.argmax(dim=2), skip_special_tokens=True), '\n',)
        # if DATA_TOKEN==False:
        #     print(label)
        # else:
        #    print(tokenizer.batch_decode(Y_ids, skip_special_tokens=True),)

        l = label_smoother(preds, lm_labels).mean()
        # l = preds.loss

        if gradient_accumulation_steps>1:
            l = l/gradient_accumulation_steps
        l.backward()
        if n % gradient_accumulation_steps == 0 or  n==steps:
            torch.nn.utils.clip_grad_norm_(net.bart.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()

        num_tokens = Y_attention_mask.sum()

        with torch.no_grad():
            metric.add(l.sum()*TRAIN_BATCH, num_tokens)

        if (n + 1) % 10 == 0:
            print(f'{n + 1} loss {metric[0] / metric[1]:.6f}, batch loss: {l.item():.6f} ')
            writer.add_scalar("train loss/token", round(metric[0] / metric[1], 6), n + epoch * len(train_dl))
    return net

def validate_batch(net, valid_dl, optimizer, tokenizer, device, use_token:bool, writer, epoch:int)->Tuple:
    net.eval()
    predictions, labels, ave_loss = [], [], 0.
    label_smoother = LabelSmoother(epsilon=LABEL_SMOOTHING)
    with torch.no_grad():
        for n, batch in tqdm(enumerate(valid_dl), total=len(valid_dl)):
            optimizer.zero_grad()

            if use_token:
                input_ids = batch["content_idx"].cuda()
                attention_mask = batch["content_mask"].cuda()
                Y_ids = batch["title_idx"][:, :-1].contiguous().cuda()
                lm_labels = batch["title_idx"][:, 1:].clone().detach().cuda()
                lm_labels[batch["title_idx"][:, 1:] == tokenizer.pad_token_id] = -100
            else:
                text = batch[0]
                label = batch[1]
                encoded = tokenizer.batch_encode_plus(
                    list(text), max_length=MAX_LEN, padding="max_length", truncation=True,
                    return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
                )
                encoded_Y = tokenizer.batch_encode_plus(
                    list(label), max_length=MAX_GEN, padding="max_length", truncation=True,
                    return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
                )
                input_ids = encoded["input_ids"].cuda()
                attention_mask = encoded["attention_mask"].cuda()
                Y_ids = encoded_Y["input_ids"][:, :-1].contiguous().cuda()
                lm_labels = encoded_Y["input_ids"][:, 1:].clone().detach().cuda()
                lm_labels[encoded_Y["input_ids"][:, 1:] == tokenizer.pad_token_id] = -100

            generate_ids = net.bart.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_GEN,
                num_beams=NUM_BEAMS,
                repetition_penalty=2.5,
                length_penalty=1.0,
                # early_stopping=True,
                # return_dict_in_generate=True,
                # output_scores=True,
                # renormalize_logits=True
            )

            with torch.no_grad():
                preds = net(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decode_input_ids=Y_ids,  # 102, 101, ...
                            label=lm_labels  # 101, ....
                            )
                loss = label_smoother(preds, lm_labels).mean().detach()
            if n%10==0:
                print("loss: ", loss)
                writer.add_scalar("val loss/batch", round(loss.item()*2 /batch["title_mask"].sum().item() , 6), n + epoch * len(valid_dl))
            ave_loss+=loss

            if generate_ids.shape[1]<MAX_GEN:
                generate_ids = torch.cat((generate_ids,
                                          torch.zeros(
                                              (generate_ids.shape[0],
                                               MAX_GEN-generate_ids.shape[1]), device=device)
                                          ), dim=1)
            predictions.append(generate_ids)

            # generate_ids = net.bart.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     max_length=MAX_GEN,
            #     num_beams=NUM_BEAMS,
            #     repetition_penalty=2.5,
            #     length_penalty=1.0,
            #     early_stopping=True,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     renormalize_logits=True
            # )
            # beam_idx_batch = generate_ids.beam_indices
            # logits_beam = [torch.exp(it[[beam_idx_batch[0][i], beam_idx_batch[1][i]], :])
            #            for i, it in enumerate(generate_ids.scores)]
            # logits = torch.zeros((Y_ids.shape[0], MAX_GEN-1, len(tokenizer)), device=try_gpu())
            # for g in range(len(logits_beam)):
            #     logits[:, g, :] = logits_beam[g]
            # print(tokenizer.batch_decode(logits.argmax(dim=2), skip_special_tokens=True), '\n',
            #       tokenizer.batch_decode(generate_ids.sequences, skip_special_tokens=True), '\n',
            #       tokenizer.batch_decode(Y_ids, skip_special_tokens=True))
            # l = criterion(logits.view(-1, len(tokenizer)), Y_ids.view(-1))
            #
            # if n%10==0:
            #     print(f"val loss is : {l.item():.6f}")
            # if generate_ids.sequences.shape[1]<MAX_GEN:
            #     generate_ids.sequences = torch.cat((generate_ids.sequences,
            #                               torch.zeros(
            #                                   (generate_ids.sequences.shape[0],
            #                                    MAX_GEN-generate_ids.sequences.shape[1]), device=device)
            #                               ), dim=1)
            # predictions.append(generate_ids.sequences)

            labels.append(Y_ids)
            # preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generate_ids]
            # target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in Y_ids]
        dic = compute_metrics((torch.cat(predictions, dim=0), torch.cat(labels, dim=0)))
    return dic, ave_loss/len(valid_dl)



def predict_BART(net, test_dl, tokenizer, use_token=False):
    net.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for n, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
            if use_token:
                input_ids = batch["content_idx"].cuda()
                attention_mask = batch["content_mask"].cuda()
            else:
                text = batch
                encoded = tokenizer.batch_encode_plus(
                    list(text), padding="max_length", max_length=MAX_LEN, truncation=True,
                    return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
                )
                input_ids = encoded["input_ids"].cuda()
                attention_mask = encoded["attention_mask"].cuda()
            generate_ids = net.bart.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_GEN,
                num_beams=NUM_BEAMS,
                # repetition_penalty=2.5,
                # length_penalty=1.0,
                # early_stopping=True
            )

            if generate_ids.shape[1] < MAX_GEN:
                generate_ids = torch.cat(
                    (
                        generate_ids, torch.zeros((generate_ids.shape[0], MAX_GEN - generate_ids.shape[1]), device=try_gpu())),
                    dim=1)
            # preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generate_ids]
            predictions.append(generate_ids)
        doc_ids = [text for bch in predictions for text in bch]
    return doc_ids

def train(train_shuffle, dev_dataset, tokenizer):
    model = None
    torch.cuda.empty_cache()
    train_dataset = train_shuffle

    tran_ds = TrainData(train_dataset, tokenizer, MAX_LEN=MAX_LEN, MAX_GEN=MAX_GEN, use_token=DATA_TOKEN)
    dev_ds = TrainData(dev_dataset, tokenizer, MAX_LEN=MAX_LEN, MAX_GEN=MAX_GEN, use_token=DATA_TOKEN)
    train_dl = DataLoader(tran_ds, batch_size=TRAIN_BATCH, shuffle=True, drop_last=False)
    dev_dl = DataLoader(dev_ds, batch_size=TRAIN_BATCH, shuffle=False, drop_last=False)

    if model != None:
        del model
    model = MyBart(BASE_MODEL_PATH).cuda()
    model.bart.resize_token_embeddings(len(tokenizer))
    model.bart.gradient_checkpointing_enable()


    trainer = Seq2SeqTrainer(model.bart, myargs,
                             train_dataset=train_dl,
                             eval_dataset=tran_ds,
                             tokenizer=tokenizer,
                             compute_metrics=compute_metrics_np)
    optimizer, scheduler = get_optimizer_scheduler(trainer, len(train_dl))
    writer = SummaryWriter(LOG_DIR + f'{time.time()}')

    valid_dl = dev_dl
    device=try_gpu()
    use_token = tran_ds.use_token
    best_loss = np.inf
    timer = Timer()
    metric = Accumulator(2)

    for epoch in range(EPOCHES):
        model.train()
        model = train_batch(model, train_dl, optimizer, tokenizer, scheduler, metric, use_token, writer, epoch)

        if (epoch + 1) % 1 == 0:
            print(f'loss {metric[0] / metric[1]:.6f}, {metric[1] / timer.stop():.1f} '
                  f'tokens/sec on {str(try_gpu())}')

        valid_dic, val_loss = validate_batch(model, valid_dl, optimizer, tokenizer, device, use_token, writer, epoch)
        # for name, value in valid_dic.items():
        print("the valid: ", [(name, value) for name, value in valid_dic.items()])

        if val_loss<best_loss:# valid_dic["rouge1"] < best_loss
            best_loss = val_loss
            print(
                f"the best rouge-1 is: {valid_dic['rouge1']}; rouge2 is {valid_dic['rouge2']}; rouge-L is {valid_dic['rougeL']}")
            # torch.save(model.state_dict(), "bert.pth")
            create_dir_not_exist(MODEL_SAVE_PATH+f"epoch{epoch}")
            model.bart.save_pretrained(MODEL_SAVE_PATH+f"epoch{epoch}")
            print("save...")
    return model



def predict(test_dataset, tokenizer, model_path, csv_save_path):
    model = MyBart(model_path).cuda()
    test_ds = TestData(test_dataset, tokenizer, use_token=DATA_TOKEN)
    test_dl = DataLoader(test_ds, batch_size=PREDICT_BATCH, shuffle=False, drop_last=False)
    pred_test = predict_BART(model, test_dl, tokenizer, use_token=DATA_TOKEN,)

    encoded_Y = tokenizer.batch_encode_plus(
        list(test_dataset["title"].values), padding="max_length", max_length=MAX_GEN, truncation=True,
        return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
    )
    input_ids = encoded_Y["input_ids"].cuda()
    pred_test = torch.cat(pred_test, dim=0).reshape(-1, MAX_GEN)

    test_dic = compute_metrics((pred_test, input_ids))
    print("the test: ", [(name, value) for name, value in test_dic.items()])
    decoded_labels = tokenizer.batch_decode(pred_test, skip_special_tokens=True)
    for i in range(len(decoded_labels)):
        test_dataset["content"].values[i] = re.sub(' ', '', decoded_labels[i])
    test_dataset.to_csv(csv_save_path, encoding="utf-8")


def crossValidation(train_shuffle, tokenizer):
    kf = KFold(n_splits=10)
    model = None
    timer = Timer()
    metric = Accumulator(2)
    for train_idx, val_idx in tqdm(kf.split(list(train_shuffle["content"]), list(train_shuffle["title"]))):
        metric.reset()
        writer = SummaryWriter(LOG_DIR + f"{time.time()}")
        torch.cuda.empty_cache()
        train_dataset = train_shuffle.loc[train_idx, :]
        val_dataset = train_shuffle.loc[val_idx, :]
        tran_ds = TrainData(train_dataset, tokenizer, MAX_LEN=MAX_LEN, MAX_GEN=MAX_GEN, use_token=True)
        valid_ds = TrainData(val_dataset, MAX_LEN=MAX_LEN, MAX_GEN=MAX_GEN, use_token=True)
        train_dl = DataLoader(tran_ds, batch_size=TRAIN_BATCH, shuffle=True, drop_last=False)
        val_dl = DataLoader(valid_ds, batch_size=TRAIN_BATCH, shuffle=False, drop_last=False)

        if model != None:
            del model, optimizer, scheduler
        model = MyBart(BASE_MODEL_PATH).cuda()
        model.bart.resize_token_embeddings(len(tokenizer))

        trainer = Seq2SeqTrainer(model.bart, myargs,
                                 train_dataset=train_dl,
                                 eval_dataset=tran_ds,
                                 tokenizer=tokenizer,
                                 compute_metrics=compute_metrics_np)
        optimizer, scheduler = get_optimizer_scheduler(trainer, len(train_dl))


        # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
        #                                             num_training_steps=len(train_dl),
        #                                             num_warmup_steps=0)


        for epoch in range(EPOCHES):
            model.train()
            label_smoother = LabelSmoother(epsilon=0.05)
            gradient_accumulation_steps = myargs.gradient_accumulation_steps
            steps = len(train_dl)
            for n, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
                optimizer.zero_grad()
                text = batch[0]
                label = batch[1]
                encoded = tokenizer.batch_encode_plus(
                    list(text), max_length=MAX_LEN, padding="max_length", truncation=True,
                    return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
                )
                encoded_Y = tokenizer.batch_encode_plus(
                    list(label), max_length=MAX_GEN, padding="max_length", truncation=True,
                    return_tensors="pt", return_attention_mask=True, return_token_type_ids=False
                )
                input_ids = encoded["input_ids"].cuda()
                attention_mask = encoded["attention_mask"].cuda()

                Y_ids = encoded_Y["input_ids"][:, :-1].contiguous().cuda()
                lm_labels = encoded_Y["input_ids"][:, 1:].clone().detach().cuda()
                lm_labels[encoded_Y["input_ids"][:, 1:] == tokenizer.pad_token_id] = -100

                Y_attention_mask = encoded_Y["attention_mask"].cuda()
                preds = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              decode_input_ids=Y_ids,
                              # decoder_attention_mask=Y_attention_mask,
                              label=lm_labels)
                l = label_smoother(preds, lm_labels).mean()
                # l = preds.loss

                if gradient_accumulation_steps > 1:
                    l = l / gradient_accumulation_steps
                l.backward()
                if n % gradient_accumulation_steps == 0 or n == steps:
                    torch.nn.utils.clip_grad_norm_(model.bart.parameters(), max_norm=1)
                    optimizer.step()
                    scheduler.step()

                num_tokens = Y_attention_mask.sum()

                with torch.no_grad():
                    metric.add(l.sum(), num_tokens)
                if (n + 1) % 10 == 0:
                    print(f'{n} loss {metric[0] / metric[1]:.6f}')
                writer.add_scalar("train loss/token", round(metric[0] / metric[1], 6), n+ epoch * len(train_dl))

            if (epoch + 1) % 1 == 0:
                print(f'loss {metric[0] / metric[1]:.6f}, {metric[1] / timer.stop():.1f} '
                      f'tokens/sec on {str(try_gpu())}')

            valid_dic, val_loss = validate_batch(model, val_dl, optimizer, tokenizer, try_gpu(), use_token=tran_ds.use_token, writer=writer, epoch=epoch)
            # for name, value in valid_dic.items():
            print("the valid: ", [(name, value) for name, value in valid_dic.items()])
            for _, value in valid_dic.items():
                if _ in ["rouge1", "rouge2", "rougeL"]:
                    writer.add_scalar(_, value, epoch)

def main():
    train_dataset, dev_dataset = data_process("train"), data_process("dev")

    TRAINING_SIZE = int(0.5*len(dev_dataset))
    dev_shuffle = dev_dataset.sample(frac=1, random_state=0)

    val_dataset = dev_shuffle[0: TRAINING_SIZE]
    test_dataset = dev_shuffle[TRAINING_SIZE:]


    # crossValidation(train_shuffle, tokenizer)
    # train(train_dataset, val_dataset, tokenizer)

    predict(dev_shuffle, tokenizer, model_path=MODEL_SAVE_PATH+"epoch1/", csv_save_path=CSV_SAVE_PATH)

if __name__ == "__main__":
    VOCAB = ["“", "”",  "’", "‘"] # , "—" "…"
    BASE_MODEL_PATH = "fnlp/bart-large-chinese"
    MODEL_SAVE_PATH = "./model/my_model/"
    LOG_DIR = "logs/train_val/"
    CSV_SAVE_PATH = "out/my_predict_1.csv"
    DATA_TOKEN = True
    MAX_LEN = 512
    MAX_GEN = 32
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.1
    LABEL_SMOOTHING = 0.05
    TRAIN_BATCH = 2
    PREDICT_BATCH = 32
    NUM_BEAMS = 4
    EPOCHES = 10
    gradient_accumulation_steps = 4
    # CV_EPOCHES = 2

    myargs = Seq2SeqTrainingArguments(
        'model',
        evaluation_strategy='steps',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=TRAIN_BATCH,
        label_smoothing_factor=LABEL_SMOOTHING,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=WEIGHT_DECAY,
        save_steps=1000,
        save_total_limit=10,
        num_train_epochs=EPOCHES,
        generation_max_length=MAX_GEN,
        predict_with_generate=True,
        generation_num_beams=NUM_BEAMS,
        eval_steps=200,
        # logging_dir=exp_dir + log_dir,
        logging_first_step=True)


    create_dir_not_exist(MODEL_SAVE_PATH)
    create_dir_not_exist(LOG_DIR)

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.add_tokens(VOCAB)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    # main()
    exp_dir = "exp_dir/"
    log_dir = 'logs2'
    create_dir_not_exist(exp_dir+log_dir)

    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    tokenizer.add_tokens(VOCAB)
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-large-chinese")
    model.resize_token_embeddings(len(tokenizer))
    batch_size = 2


    args = Seq2SeqTrainingArguments(
        exp_dir+'model',
        evaluation_strategy='steps',
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        label_smoothing_factor=0.05,
        gradient_accumulation_steps=4,
        weight_decay=0.1,
        save_steps=200,
        save_total_limit=25,
        num_train_epochs=5,
        generation_max_length=32,
        predict_with_generate=True,
        generation_num_beams=4,
        eval_steps=200,
        # gradient_checkpointing=True,
        logging_dir=exp_dir + log_dir,
        logging_first_step=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding=True)
    data = Data(tokenizer)
    tokenized_train_dataset, tokenized_test_dataset = data.preProcess()



    trainer = Seq2SeqTrainer(model, args,
                             train_dataset=tokenized_train_dataset,
                             eval_dataset=tokenized_test_dataset,
                             data_collator=data_collator,
                             tokenizer=tokenizer,
                             compute_metrics=compute_metrics_np)




    tmp = trainer.train()
    print(tmp[2]["train_loss"])

    print(tmp)

