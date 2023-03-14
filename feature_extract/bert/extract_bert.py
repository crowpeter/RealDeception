#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:13:34 2020

@author: shaohao
"""
#%% init
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from glob import glob
import os
import joblib

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"]="0"
MODE = 'single_binary_label'
epochs = 15
batch_size = 6

if MODE == 'single_origin_label':
    SIZE=3
    DATA_FILE = "/homes/GPU2/shaohao/bert_aac/Data/ALL_DATA.pkl"
    
elif MODE == 'conv_origin_label':
    SIZE=3
    DATA_FILE = "/homes/GPU2/shaohao/bert_aac/Data/ALL_DATA_conversation.pkl"
    
elif MODE == 'single_binary_label':
    SIZE=2
    DATA_FILE = "/homes/GPU2/shaohao/bert_aac//Data/ALL_DATA_binary.pkl"
    
elif MODE == 'conv_binary_label':
    SIZE=2
    DATA_FILE = "/homes/GPU2/shaohao/bert_aac/Data/ALL_DATA_conversation_binary.pkl"

PRETRAINED_MODEL_NAME = "bert-base-chinese"
LEARNING_RATE = 2e-3

# specify GPU device
# os.environ["CUDA_VISIBLE_DEVICES"] ="1"
# torch.cuda.set_device(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

torch.cuda.empty_cache()


#%% data preprocess

print("Preparing DATA: ", DATA_FILE)
all_df = joblib.load(DATA_FILE)
        
all_text = all_df.loc[:,['content']].values.tolist()
all_label = torch.tensor(all_df.loc[:,['label']].values.astype(int))

for ind, text in enumerate(all_text):
    if len(text[0]) > 500:
        # print(text[0])
        # print('amswer: ', text[0].split('??)[-1])
        if len(text[0].split('。')[-1]) > 500:
            # print('sol1')
            output = text[0].split('。')[-1][:500]
        else:
            answer = text[0].split('。')[-1]
            output = '，'.join(text[0].split('。')[0].split('，')[-10:]) + '。' + answer

            
        # print('out: ', text[0])
        # print(len(text[0]))
        # print()
    else:
        output = text[0]
    # if output.count('??)>1:
    #     print(ind, text)
    temp = ['[CLS] ' + output + ' [SEP]']
    tokens = tokenizer.tokenize(temp[0])
    all_text[ind] = tokenizer.convert_tokens_to_ids(tokens)