#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:39:56 2023

@author: aac
"""

import torch
import argparse
import glob
import numpy as np

from utils import check_load_fea
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
from loaders.multitask_loader import aac_concate_speech_fusion_Dataset_multitask_analysis
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

import os
import random
import joblib
# import pdb
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#%%


def main(args, thresh):
    # pdb.set_trace()
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
    
    FEATURE_ROOT_PATH_LIST = []
    FEAT_DIM_BOX = []
    if 'bhv' in args.feature:
        if args.conversation:
            FEAT_DIM_BOX.append(14)
        else:
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
    
    FEATURE = args.feature.split('+')
    EXP_PATH = args.save_path+args.feature+'_'+args.model+'/'
    if not os.path.exists(EXP_PATH):
        os.mkdir(EXP_PATH)
    time_step = args.wrap # fix
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    final_box_path = EXP_PATH
    
    #%%
    # 33 ep
    eps = sorted(glob.glob('/homes/GPU0/aac/speech_feature_extraction/emobase_pkl/*')) ## be careful
    eps = [os.path.basename(i) for i in eps]
    non_dece_set = eps.copy()
    
    # load meta label
    eps_feapath_label, non_dece_set = check_load_fea(non_dece_set, FEATURE_ROOT_PATH_LIST, args.meta_path)
    cv_ts_box = joblib.load("CVBOX/CV.pkl")
    # pdb.set_trace()
    
    #%% training main
    FIANL_BOX={
        'name':[],
        'out_prob':[],
        'out_prob_mil':[],
        'pred_final':[],
        'true_final':[],
        'pred_final_mil':[],
        'true_final_mil':[],
        'correct_name':[],
        'correct_cv_idx':[]
        }
    activation = {}
    for cv_idx, (ts_ep_keys, tr_ep_keys) in enumerate(cv_ts_box):
        ts_dic = {}
        for ts_key in ts_ep_keys:
            for (k,v) in  eps_feapath_label[ts_key].items():
                ts_dic[str(ts_key)+'/'+str(k)+'_a.pkl'] = v
    
        model_path = "BEST_MODEL/{}/{}/conv_{}_cv_{}_timestep_{}_mil_{}_alpha_0.5.sav".format(args.model,'_'.join(FEATURE), args.conversation, cv_idx, time_step, args.mil_mode)
        # pdb.set_trace()
        print(model_path)
        model = torch.load(model_path)
        model.eval()
        with torch.no_grad():
            # ts_datasets = aac_speech_fusion_Dataset_multitask_analysis(ep_keys = ts_dic, eps_path = eps_feapath_label, feature_root = FEATURE_ROOT_PATH_LIST, wrap_len = args.wrap, conversation = args.conversation, func = args.functional, label_weighted=True)
            ts_datasets = aac_concate_speech_fusion_Dataset_multitask_analysis(ep_keys = ts_dic, eps_path = eps_feapath_label, feature_root = FEATURE_ROOT_PATH_LIST, wrap_len = args.wrap, conversation = args.conversation, func = args.functional)

            ts_loaders = DataLoader(ts_datasets, batch_size=len(ts_datasets))            
            
            for step, (batch_x, batch_y, mil_y, name) in enumerate(ts_loaders):
                # pdb.set_trace()
                name = pd.DataFrame(name).T
                name['add'] = name.apply(lambda x: [x[i] for i in range(time_step)], axis=1)
                name = list(name['add'])
                # name = [x for xs in name for x in xs]
                FIANL_BOX['name'].extend(name)
                # pdb.set_trace()
                batch_x = [i.float().to(device) for i in batch_x]
                target = batch_y.float().to(device) 
                target_mil = mil_y.float().to(device) 
                
                seq_len = torch.tensor([time_step for i in range(len(batch_x[0]))]).to(device)
                # pdb.set_trace()
                
                def get_activation(name, cv_idx):
                    def hook(model, input, output):
                        # pdb.set_trace()
                        output, output_mil = output
                        activation[name+"_CV{}".format(cv_idx)] = output.detach()
                    return hook
                
                if 'BiLSTM' in model_path:
                    output, output_mil = model.forward(batch_x)
                    output = output.squeeze(dim=2)
                elif 'TRANS' in model_path:
                    output, output_mil = model.forward(batch_x, seq_len)
                    output = torch.cat(output, dim=1).T
                output_mil = output_mil.reshape(-1,1)
                # pdb.set_trace()

                output[output>=thresh]=1
                output[output<thresh]=0
                pred = output
                
                output_mil[output_mil>=thresh]=1
                output_mil[output_mil<thresh]=0
                pred_mil = output_mil
                
                pred_dd = pred.data.cpu().numpy().reshape(-1,).tolist()
                target_dd = target.data.cpu().numpy().reshape(-1,).tolist()
                
                pred_mil_dd = pred_mil.data.cpu().numpy().reshape(-1,).tolist()
                target_mil_dd = target_mil.data.cpu().numpy().reshape(-1,).tolist()
                FIANL_BOX['pred_final'].extend(pred_dd)
                FIANL_BOX['true_final'].extend(target_dd)
                FIANL_BOX['pred_final_mil'].extend(pred_mil_dd)
                FIANL_BOX['true_final_mil'].extend(target_mil_dd)
                # pdb.set_trace()
                
                for i in range(len(name)):
                    if target_mil_dd[i]==1 and target_mil_dd[i] == pred_mil_dd[i]:
                        
                        FIANL_BOX['correct_name'].append(name[i])
                        FIANL_BOX['correct_cv_idx'].append(str(cv_idx)+'-'+str(i))
        
    precision = precision_score(FIANL_BOX['true_final'], FIANL_BOX['pred_final'], average='macro', zero_division=0)
    UAR = recall_score(FIANL_BOX['true_final'], FIANL_BOX['pred_final'], average='macro', zero_division=0)
    f1score = f1_score(FIANL_BOX['true_final'], FIANL_BOX['pred_final'], average='macro', zero_division=0)
    ACC = accuracy_score(FIANL_BOX['true_final'], FIANL_BOX['pred_final'])
    
    print('=============================\n')
    print('features: '+ args.feature + 'thresh: {}'.format(thresh) +'\n'+
          ' UAR: {:.4f}  ACC: {:.4f}  F1 score: {:.4f} Precision: {:.4f}'.format(UAR, ACC, f1score, precision))
    print('=============================\n')
    print('total correct context number:', len(FIANL_BOX['correct_name']))
    joblib.dump(FIANL_BOX, os.path.join(final_box_path, 'MIL_conv_{}_timestep_{}_mil_{}_thresh_{}.pkl'.format(args.conversation, time_step, args.mil_mode, thresh)))
    
    return FIANL_BOX


   
#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default= 'bert_pkl+emobase_pkl+bhv_feature', help='input data feature set')
    parser.add_argument('--meta_path', type=str, default="/homes/GPU2/shaohao/bert_aac/BERT_emb/",help='meta pkl dir')
    parser.add_argument('--conversation', type=str, default='yes', help='input data mode')
    parser.add_argument('--model', type=str,  default="TRANS_multitask_concat_fea",help='model select')
    parser.add_argument('--functional', type=str,  default='mean', help='feature functional')
    parser.add_argument('--wrap', type=int, default=15, help='wrap how many sentence one time')
    parser.add_argument('--mil_mode', type=str, default='A-self_att', help='padding')
    parser.add_argument('--save_path', type=str, default="inference_test/", help='lstm layer dim')
    
    args = parser.parse_args()
    if args.conversation == 'yes':
        args.conversation = True
    else:
         args.conversation = False
    
    final_box = main(args,0.5)
#%% 

    # recall_lst = []
    # f1_lst = []
    
    # recall_lst_mil = []
    # f1_lst_mil = []
    # for t in range(0,105, 5):
    #     pred = final_box['out_prob']
    #     pred = np.array(pred)
    #     pred[pred>=(t/100.0)] = 1
    #     pred[pred<(t/100.0)] = 0
        
    #     recall = recall_score(final_box['true_final'], pred, average='macro', zero_division=0)
    #     f1score = f1_score(final_box['true_final'], pred, average='macro', zero_division=0)
        
    #     recall_lst.append(recall)
    #     f1_lst.append(f1score)
    #     ##############################
    #     pred = final_box['out_prob_mil']
    #     pred = np.array(pred)
    #     pred[pred>=(t/100.0)] = 1
    #     pred[pred<(t/100.0)] = 0
        
    #     recall = recall_score(final_box['true_final_mil'], pred, average='macro', zero_division=0)
    #     f1score = f1_score(final_box['true_final_mil'], pred, average='macro', zero_division=0)
        
    #     recall_lst_mil.append(recall)
    #     f1_lst_mil.append(f1score)
        
    # result = pd.DataFrame()
    # result['thresh'] = [i/100.0 for i in range(0,105,5)]
    # result['UAR'] = recall_lst
    # result['f1score'] = f1_lst
    # result['UAR_mil'] = recall_lst_mil
    # result['f1score_mil'] = f1_lst_mil
    # result.to_csv('thresh_UAR_F1.csv')
    
    # plt.figure()
    # plt.plot([i for i in range(len(recall_lst))], recall_lst, label = 'recall')
    # plt.plot([i for i in range(len(f1_lst))], f1_lst, label = 'f1')
    # plt.xlabel('thresh')
    # plt.ylabel('UAR')
    # plt.legend()
    # # plt.savefig('recall_f1.png')
    # plt.show()    
        
        
        
        
        
        
    