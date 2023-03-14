#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:12:16 2022

@author: shaohao
"""

import joblib
import torch
import argparse
import glob
import math
from os.path import exists
import numpy as np
# import random
# from sklearn.model_selection import KFold
from utils import BatchSampler, check_load_fea, manual_fold_cv
import tqdm
from models import multi_input_fusion_TFM_multitask_FBP, multi_input_fusion_Bilstm_multitask
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score, precision_score
from loaders.multitask_loader import aac_speech_fusion_Dataset_multitask, aac_concate_speech_fusion_Dataset_multitask
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import random
import pdb
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# pdb.set_trace()

parser = argparse.ArgumentParser()
parser.add_argument('--feature', type=str, default= 'bert_pkl+emobase_pkl+bhv_feature', help='input data feature set')
parser.add_argument('--meta_path', type=str, default="/homes/GPU2/shaohao/bert_aac/BERT_emb/",help='meta pkl dir')
parser.add_argument('--conversation', type=str, default='yes', help='input data mode')
parser.add_argument('--model', type=str,  default="TRANS",help='model select')
parser.add_argument('--learning_rate', type=float,  default=1e-3,help='learning rate')
parser.add_argument('--functional', type=str,  default='mean', help='feature functional')
parser.add_argument('--batch_size', type=int,  default=64, help='batch size')
parser.add_argument('--node_num', type=int,  default=128, help='batch size')
parser.add_argument('--layer_num', type=int,  default=3, help='batch size')
parser.add_argument('--loss', type=str, default='CE', help='padding')
parser.add_argument('--epoch', type=int, default=30, help='padding')
parser.add_argument('--wrap', type=int, default=18, help='wrap how many sentence one time')
parser.add_argument('--alpha', type=float, default=0.3, help='wrap how many sentence one time')
parser.add_argument('--mil_mode', type=str, default='Q-self_att', help='padding')

args = parser.parse_args()
if args.conversation == 'yes':
    args.conversation = True
else:
     args.conversation = False

# pdb.set_trace()
#%%
# warnings.filterwarnings("ignore")
FOLD=5
RANDSEED = 2021
np.random.seed(RANDSEED)
torch.manual_seed(RANDSEED)
torch.cuda.manual_seed(RANDSEED)
torch.cuda.manual_seed_all(RANDSEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(RANDSEED)
random.seed(RANDSEED)
output_result_path = './result_{}_multitask.csv'.format(args.model)
#%%
FEATURE_ROOT_PATH_LIST = []
FEAT_DIM_BOX = []
if 'bhv' in args.feature:
    FEAT_DIM_BOX.append(7)
    FEATURE_ROOT_PATH_LIST.append('/homes/GPU0/aac/behavior_feature_extraction/bhv_feature/')
if 'wav2vec2' in args.feature:
    FEAT_DIM_BOX.append(512)
    FEATURE_ROOT_PATH_LIST.append('/homes/GPU0/aac/speech_feature_extraction/wav2vec2/')
if 'bert_pkl' in args.feature:
    FEAT_DIM_BOX.append(768)
    FEATURE_ROOT_PATH_LIST.append('/homes/GPU0/aac/speech_feature_extraction/bert_pkl/')
if 'emobase_pkl' in args.feature:
    FEAT_DIM_BOX.append(988)
    FEATURE_ROOT_PATH_LIST.append('/homes/GPU0/aac/speech_feature_extraction/emobase_pkl/')
    
if args.conversation:
    FEAT_DIM_BOX = [int(ele*2) for ele in FEAT_DIM_BOX]
CLASS_NUM = 1
# NODE_NUM = [512, 128, 128, CLASS_NUM]
NODE_NUM = [args.node_num for i in range(args.layer_num)] + [CLASS_NUM]
BATCH_NORM = False
LR = args.learning_rate
time_step = args.wrap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fea_path_output = '_'.join(args.feature.split('+'))
MODEL_SAVE = 'BEST_MODEL/{}_multitask/{}/'.format(args.model, fea_path_output)
# CV_SAVE = 'CVBOX/{}/'.format(args.model)
if not os.path.exists(MODEL_SAVE):
    os.mkdir(MODEL_SAVE)
# device = torch.device("cpu")
# pdb.set_trace()
#%%
# 33 ep
eps = sorted(glob.glob('/homes/GPU0/aac/speech_feature_extraction/emobase_pkl/*'))
eps = [os.path.basename(i) for i in eps]
eps_feapath_label, non_dece_set = check_load_fea(eps, FEATURE_ROOT_PATH_LIST, args.meta_path)
# cv_ts_box = manual_fold_cv(FOLD, eps, non_dece_set)
# joblib.dump(cv_ts_box, os.path.join(CV_SAVE, 'conv_{}_{}.cv'.format(args.conversation,fea_path_output)))
cv_ts_box = joblib.load('CVBOX/CV.pkl')
# pdb.set_trace()
#%% training main
FIANL_BOX={
    'pred_final_indiv':[],
    'true_final_indiv':[],
    'pred_final_mil':[],
    'true_final_mil':[],
    }
for cv_idx, (ts_ep_keys, tr_ep_keys) in enumerate(cv_ts_box):
    model_save_path = os.path.join(MODEL_SAVE, 'conv_{}_cv_{}_timestep_{}_mil_{}.sav'.format(args.conversation,cv_idx, time_step, args.mil_mode))
    RE_BOX={
        'tr_epoch_loss':[],
        'ts_epoch_loss':[],
        'ts_epoch_UAR':[],
        'ts_epoch_ACC':[],
        'pred_indiv':[],
        'true_indiv':[],
        'pred_mil':[],
        'true_mil':[]
        }
    tr_dic = {}
    for tr_key in tr_ep_keys:
        for (k,v) in  eps_feapath_label[tr_key].items():
            tr_dic[str(tr_key)+'/'+str(k)+'_a.pkl'] = v
    ts_dic = {}
    for ts_key in ts_ep_keys:
        for (k,v) in  eps_feapath_label[ts_key].items():
            ts_dic[str(ts_key)+'/'+str(k)+'_a.pkl'] = v
    if args.model == 'BiLSTM':
        model = multi_input_fusion_Bilstm_multitask(FEAT_DIM_BOX, NODE_NUM, time_step, args.mil_mode, device).to(device)
    elif args.model == 'TRANS':
        model = multi_input_fusion_TFM_multitask_FBP(FEAT_DIM_BOX, NODE_NUM, time_step, args.mil_mode, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    for epoch in tqdm.tqdm(range(args.epoch)):

        tr_datasets = aac_concate_speech_fusion_Dataset_multitask(ep_keys = tr_dic, eps_path = eps_feapath_label, feature_root = FEATURE_ROOT_PATH_LIST, wrap_len = args.wrap, conversation = args.conversation, func = args.functional, label_weighted=True)
        weight = WeightedRandomSampler(tr_datasets.data_weight, len(tr_datasets))
        tr_loaders = DataLoader(tr_datasets, sampler=weight, batch_size=args.batch_size)

        loss_batch_record = []
        model.train()
        for step, (batch_x, batch_y_indiv, batch_y_mil) in enumerate(tr_loaders):
            input_x = [i.float().to(device) for i in batch_x]
            optimizer.zero_grad()
            target_indiv = batch_y_indiv.float().to(device)   
            target_mil = batch_y_mil.float().squeeze(2).to(device)   
            seq_len = torch.tensor([time_step for i in range(len(batch_x[0]))]).to(device)
            if args.model == 'BiLSTM':
                output_indiv, output_mil = model.forward(input_x)
                output_indiv = output_indiv.squeeze(dim=2)
            elif args.model == 'TRANS':
                output_indiv, output_mil = model.forward(input_x, seq_len)
                output_indiv = torch.cat(output_indiv, dim=1).T
            output_mil = output_mil.reshape(-1,1)
            
            loss_indiv = criterion(output_indiv, target_indiv)
            loss_mil = criterion(output_mil, target_mil)
            loss = args.alpha * loss_indiv + (1-args.alpha) * loss_mil
            # pdb.set_trace()
            loss.backward()
            optimizer.step()
            loss_batch_record.append(loss.data.cpu().numpy())
            if (step+1) % 50 == 0:
                print(' cv: '+str(cv_idx)+' epoch: '+str(epoch)+' step '+str(step+1)+\
                      ' loss: '+ str(loss.data.cpu().numpy()))
        RE_BOX['tr_epoch_loss'].append(sum(loss_batch_record)/(step+1))
        loss_batch_record = []
        with torch.no_grad():
            # aaa
            model.eval()
            best_val_loss = np.inf
            ts_datasets = aac_speech_fusion_Dataset_multitask(ep_keys = ts_dic, eps_path = eps_feapath_label, feature_root = FEATURE_ROOT_PATH_LIST, wrap_len = args.wrap, conversation = args.conversation, func = args.functional, label_weighted=True)
            ts_loaders = DataLoader(ts_datasets, batch_size=len(ts_datasets))    
            
            for step, (batch_x, batch_y_indiv, batch_y_mil) in enumerate(ts_loaders):
                input_x = [i.float().to(device) for i in batch_x]
                optimizer.zero_grad()
                target_indiv = batch_y_indiv.float().to(device)   
                target_mil = batch_y_mil.float().squeeze(2).to(device)   
                seq_len = torch.tensor([time_step for i in range(len(batch_x[0]))]).to(device)
                if args.model == 'BiLSTM':
                    output_indiv, output_mil = model.forward(input_x)
                    output_indiv = output_indiv.squeeze(dim=2)
                elif args.model == 'TRANS':
                    output_indiv, output_mil = model.forward(input_x, seq_len)
                    output_indiv = torch.cat(output_indiv, dim=1).T
                #output_mil = output_mil.reshape(-1,1)
                
                loss_indiv = criterion(output_indiv, target_indiv)
                loss_mil = criterion(output_mil, target_mil)
                loss = args.alpha * loss_indiv + (1-args.alpha) * loss_mil
                # pdb.set_trace()
            avg_loss = loss/len(ts_datasets)
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(model, model_save_path)
            
            output_indiv[output_indiv>=0.5]=1
            output_indiv[output_indiv<0.5]=0
            
            output_mil[output_mil>=0.5]=1
            output_mil[output_mil<0.5]=0
            pred_indiv = output_indiv
            pred_mil = output_mil
            
            RE_BOX['pred_indiv'].extend(pred_indiv.data.cpu().numpy().reshape(-1,).tolist())
            RE_BOX['true_indiv'].extend(target_indiv.data.cpu().numpy().reshape(-1,).tolist())
            RE_BOX['pred_mil'].extend(pred_mil.data.cpu().numpy().reshape(-1,).tolist())
            RE_BOX['true_mil'].extend(target_mil.data.cpu().numpy().reshape(-1,).tolist())

            loss_batch_record.append(loss.data.cpu().numpy())
            RE_BOX['ts_epoch_loss'].append(sum(loss_batch_record)/(step+1))
            
    #%%
    loss_batch_record = []
    model = torch.load(model_save_path)
    with torch.no_grad():
        model.eval()
        ts_datasets = aac_speech_fusion_Dataset_multitask(ep_keys = ts_dic, eps_path = eps_feapath_label, feature_root = FEATURE_ROOT_PATH_LIST, wrap_len = args.wrap, conversation = args.conversation, func = args.functional, label_weighted=True)
        ts_loaders = DataLoader(ts_datasets, batch_size=len(ts_datasets))    
        for step, (batch_x, batch_y_indiv, batch_y_mil) in enumerate(ts_loaders):
            input_x = [i.float().to(device) for i in batch_x]
            optimizer.zero_grad()
            target_indiv = batch_y_indiv.float().to(device)   
            target_mil = batch_y_mil.float().squeeze(2).to(device)   
            seq_len = torch.tensor([time_step for i in range(len(batch_x[0]))]).to(device)
            if args.model == 'BiLSTM':
                output_indiv, output_mil = model.forward(input_x)
                output_indiv = output_indiv.squeeze(dim=2)
            elif args.model == 'TRANS':
                output_indiv, output_mil = model.forward(input_x, seq_len)
                output_indiv = torch.cat(output_indiv, dim=1).T
            output_mil = output_mil.reshape(-1,1)
            
            loss_indiv = criterion(output_indiv, target_indiv)
            loss_mil = criterion(output_mil, target_mil)
            loss = args.alpha * loss_indiv + (1-args.alpha) * loss_mil
                
            output_indiv[output_indiv>=0.5]=1
            output_indiv[output_indiv<0.5]=0
            
            output_mil[output_mil>=0.5]=1
            output_mil[output_mil<0.5]=0
            pred_indiv = output_indiv
            pred_mil = output_mil
            # pdb.set_trace()
            FIANL_BOX['pred_final_indiv'].extend(pred_indiv.data.cpu().numpy().reshape(-1,).tolist())
            FIANL_BOX['true_final_indiv'].extend(target_indiv.data.cpu().numpy().reshape(-1,).tolist())
            FIANL_BOX['pred_final_mil'].extend(pred_mil.data.cpu().numpy().reshape(-1,).tolist())
            FIANL_BOX['true_final_mil'].extend(target_mil.data.cpu().numpy().reshape(-1,).tolist())

            
            loss_batch_record.append(loss.data.cpu().numpy())
        RE_BOX['ts_epoch_loss'].append(sum(loss_batch_record)/(step+1))
        UAR_indiv = recall_score(FIANL_BOX['true_final_indiv'], FIANL_BOX['pred_final_indiv'], average='macro')
        ACC_indiv = accuracy_score(FIANL_BOX['true_final_indiv'], FIANL_BOX['pred_final_indiv'])
        CM_indiv = confusion_matrix(FIANL_BOX['true_final_indiv'], FIANL_BOX['pred_final_indiv'])
        
        UAR_mil = recall_score(FIANL_BOX['true_final_mil'], FIANL_BOX['pred_final_mil'], average='macro')
        ACC_mil = accuracy_score(FIANL_BOX['true_final_mil'], FIANL_BOX['pred_final_mil'])
        CM_mil = confusion_matrix(FIANL_BOX['true_final_mil'], FIANL_BOX['pred_final_mil'])
        print('================ TEST CV {}/5 ====================='.format(cv_idx+1))
        print(cv_idx, 'indiv UAR: {:.4f}, MIL UAR: {:.4f}'.format(UAR_indiv, UAR_mil))

    RE_BOX['UAR_indiv'] = UAR_indiv
    RE_BOX['ACC_indiv'] = ACC_indiv
    
    RE_BOX['UAR_indiv'] = UAR_mil
    RE_BOX['ACC_indiv'] = ACC_mil
    FIANL_BOX['CV'+str(cv_idx)] = RE_BOX
    
UAR_indiv = recall_score(FIANL_BOX['true_final_indiv'], FIANL_BOX['pred_final_indiv'], average='macro', zero_division = 0)
precision_indiv = precision_score(FIANL_BOX['true_final_indiv'], FIANL_BOX['pred_final_indiv'], average='macro', zero_division = 0)
f1score_indiv = f1_score(FIANL_BOX['true_final_indiv'], FIANL_BOX['pred_final_indiv'], average='macro', zero_division = 0)
ACC_indiv = accuracy_score(FIANL_BOX['true_final_indiv'], FIANL_BOX['pred_final_indiv'])

UAR_mil = recall_score(FIANL_BOX['true_final_mil'], FIANL_BOX['pred_final_mil'], average='macro', zero_division = 0)
precision_mil = precision_score(FIANL_BOX['true_final_mil'], FIANL_BOX['pred_final_mil'], average='macro', zero_division = 0)
f1score_mil = f1_score(FIANL_BOX['true_final_mil'], FIANL_BOX['pred_final_mil'], average='macro', zero_division = 0)
ACC_mil = accuracy_score(FIANL_BOX['true_final_mil'], FIANL_BOX['pred_final_mil'])

print('=============================\n')
print('features: '+ args.feature +' conversation: '+str(args.conversation)+'\n'+
      ' indiv UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f} Precision: {:.4f}'.format(UAR_indiv, ACC_indiv, f1score_indiv, precision_indiv))
print(' multi UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f} Precision: {:.4f}'.format(UAR_mil, ACC_mil, f1score_mil, precision_mil))
print('=============================\n')
