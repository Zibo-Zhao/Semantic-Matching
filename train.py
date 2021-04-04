import config
import pandas as pd
from sklearn.model_selection import train_test_split
import dataset
import torch
from torch.utils.data import Dataset, DataLoader
from model import BERT_wmm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import training
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import json
import copy

def run():

    vocab = pd.read_csv(config.VOCAB_PATH, names=['word'])
    vocab_dict = {}
    for key, value in vocab.word.to_dict().items():
        vocab_dict[value] = key

    with open(config.ORIGINAL_VOCAB_PATH, 'r', encoding="UTF-8") as f:
        lines = f.read()
        tokens = lines.split('\n')
    token_dict = dict(zip(tokens, range(len(tokens))))
    counts = json.load(open(config.COUNT_PATH))
    del counts['[CLS]']
    del counts['[SEP]']
    freqs = [
        counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
    ]
    keep_tokens = list(np.argsort(freqs)[::-1])
    keep_tokens = [0, 100, 101, 102, 103, 100, 100] + keep_tokens[:(len(vocab_dict)-7)]



    train_all = pd.read_csv('./data/gaiic_track3_round1_train_20210228.tsv', sep='\t', names=['q1', 'q2', 'label'])
    test = pd.read_csv('data/gaiic_track3_round1_testA_20210228.tsv', sep='\t', names=['q1', 'q2', 'label'])

    train, valid=train_test_split(train_all, test_size=0.2, random_state=5 )

    #  train=train.reset_index(drop=True)
    # valid = valid.reset_index(drop=True)

    # create fake labels for protected groups
    temp = pd.concat([test, valid])
    temp.label = -5
    train= pd.concat([train, temp])

    train=train
    valid=valid

    train=train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    


    train_dataset=dataset.BERTDataset(train['q1'], train['q2'], train['label'])
    valid_dataset = dataset.BERTDataset(valid['q1'], valid['q2'], valid['label'])
    # test_dataset = dataset.BERTDataset(test['q1'], test['q2'], test['label'])

    train_data_loader=DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=4)
    valid_data_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, num_workers=4)
    # test_data_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=4)

    print(len(train_data_loader))
    model=BERT_wmm(keep_tokens)
    model_teacher = BERT_wmm(keep_tokens)

    device=torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
    print(device)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)


    best_auc_score=0.5
    model=nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    model.to(device)
    model_teacher = nn.DataParallel(model_teacher, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    model_teacher.to(device)

    parameter_list=[]
    Distillation_flag = False
    for epoch in range(config.EPOCHS):
        training.train_fn(train_data_loader, model, optim, device, model_teacher, Distillation_flag)
        sdA=model.state_dict()
        if len(parameter_list)<config.K:
            parameter_list.append(sdA)
        else:
            parameter_list.pop(0)
            parameter_list.append(sdA)
            Distillation_flag=True
            sdB=parameter_list[0]
            for key in parameter_list[0]:
                sdB[key]=sdB[key]/float(config.K)
                for i in range(1, len(parameter_list)):
                    sdB[key]=sdB[key]+(parameter_list[i][key]/float(config.K))
            model_teacher.load_state_dict(sdB)

        if Distillation_flag:
            outputs, targets=training.eval_fn(valid_data_loader, model_teacher, device)
        else:
            outputs, targets = training.eval_fn(valid_data_loader, model, device)
        auc_scores = roc_auc_score(targets, outputs)
        print('---------------------------------------------')
        print("auc_score{}".format(auc_scores))
        print('---------------------------------------------')
        if auc_scores>best_auc_score:
            best_auc_score=auc_scores
    print(best_auc_score)

run()

