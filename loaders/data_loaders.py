#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:37:56 2022

@author: shaohao
"""

import joblib
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence 
import torch
import sys
import pandas as pd
import glob
from collections import Counter
# from torchsampler import ImbalancedDatasetSampler
import random
import os
import pdb
'''
class aac_speech_bypath_Dataset(Dataset):
    def __init__(self, feature_dict, feature_name, conversation=False, func=None, label_weighted=False):
        self._X = list(feature_dict.keys())
        random.shuffle(self._X)
        self._Y = [feature_dict[key] for key in self._X]
        self.fea_name = feature_name
        self.func = func
        self.conversation = conversation
        assert len(self._X) == len(self._Y)
        if label_weighted:
            self.data_weight = []
            # self.label_weight = [1, 1]
            self.label_weight = [1/item[1] for item in sorted(Counter(self._Y).items())]
            for i in self._Y:
                if i == 0:
                    self.data_weight.append(self.label_weight[0])
                elif i == 1:
                    self.data_weight.append(self.label_weight[1])
    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        x_data = joblib.load(self._X[idx])
        # print(self._X[idx])
        if self.fea_name == 'emobase_pkl':
            x_data = x_data
        if self.fea_name == 'wav2vec2':
            # try:
            if len(x_data.shape) == 1:
                x_data = np.expand_dims(x_data, axis=0)
            if self.func == 'mean':
                x_data = np.mean(x_data,axis=0)
            elif self.func == 'max':
                x_data = np.max(x_data,axis=0)
        elif self.fea_name == 'bert_pkl':
            x_data = x_data
            if len(x_data.shape) == 1:
                x_data = np.expand_dims(x_data, axis=0)
            if self.func == 'mean':
                x_data = np.mean(x_data,axis=0)
            elif self.func == 'max':
                x_data = np.max(x_data,axis=0)
            
        if self.conversation:
            # print(self._X[idx].replace('_a', '_q'))
            x_data_q = joblib.load(self._X[idx].replace('_a', '_q'))
            if self.fea_name == 'wav2vec2':
                if len(x_data_q.shape) == 1:
                    x_data_q = np.expand_dims(x_data_q, axis=0)
                if self.func == 'mean':
                    x_data_q = np.mean(x_data_q,axis=0)
                elif self.func == 'max':
                    x_data_q = np.max(x_data_q,axis=0)
            elif self.fea_name == 'emobase_pkl':
                x_data_q = x_data_q
            x_data = np.mean(np.array([x_data, x_data_q]),axis=0)
            
        y_data = self._Y[idx]
        return x_data, y_data
'''
class aac_speech_fusion_Dataset(Dataset):
    def __init__(self, ep_keys, eps_path, feature_root, wrap_len, conversation=False, func=None, label_weighted=False):
        
        self.wrap_len = wrap_len
        all_key = list(ep_keys.keys())
        if wrap_len != 1:
            output = []
            temp = []
            current_recording = all_key[0].split('/')[0]
            for i in range(len(all_key)):
                if len(temp) != wrap_len and current_recording == all_key[i].split('/')[0]:
                    temp.append(all_key[i])
                else:
                    if len(temp)==self.wrap_len:
                        output.append(temp.copy())
                    temp.clear()
                current_recording = all_key[i].split('/')[0]
                    
            self._X = output
            # random.shuffle(self._X)
            self._Y = []
            for i in self._X:
                self._Y.append([ep_keys[key] for key in i])
            flat_list = [1 if 1 in sublist else 0 for sublist in self._Y ]
            # pdb.set_trace()
        else:
            self._X = all_key
            random.shuffle(self._X)
            self._Y = [ep_keys[key] for key in self._X]
            flat_list = self._Y
        self.func = func
        self.feature_root = feature_root
        # pdb.set_trace()
        self.conversation = conversation
        assert len(self._X) == len(self._Y)
        # pdb.set_trace()
        if label_weighted:
            self.data_weight = []
            # self.label_weight = [1, 1]
            self.label_weight = [1/item[1] for item in sorted(Counter(flat_list).items())]
            for i in self._Y:
                if self.wrap_len==1:
                    if i == 0:
                        self.data_weight.append(self.label_weight[0])
                    elif i == 1:
                        self.data_weight.append(self.label_weight[1])
                else:
                    if 1 in i:
                        self.data_weight.append(self.label_weight[1])
                    else:
                        self.data_weight.append(self.label_weight[0])
        # pdb.set_trace()
                    
    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        output_data = []
        
        for fea in self.feature_root:
            
            if self.wrap_len == 1:
                data = joblib.load(os.path.join(fea, self._X[idx]))
                current_fea = fea.split('/')[-2]
                if current_fea == 'bhv_feature':    
                    data = np.array(data)
                    
                elif current_fea == 'wav2vec2':
                    if len(data.shape) == 1:
                        data = np.expand_dims(data, axis=0)
                    if self.func == 'mean':
                        data = np.mean(data,axis=0)
                    elif self.func == 'max':
                        data = np.max(data,axis=0)
                        
                elif current_fea == 'bert_pkl':
                    if len(data.shape) == 1:
                        data = np.expand_dims(data, axis=0)
                    if self.func == 'mean':
                        data = np.mean(data,axis=0)
                    elif self.func == 'max':
                        data = np.max(data,axis=0)
                
                elif current_fea == 'emobase_pkl':
                    data = data
                    
                
                if self.conversation:
                    data_q = joblib.load(os.path.join(fea, self._X[idx]).replace('_a', '_q'))
                    # import pdb;pdb.set_trace()
                    if current_fea == 'bhv_feature':    
                        data_q = np.array(data_q)
                        data = np.hstack((data, data_q))
                    else:
                        if current_fea == 'wav2vec2':
                            if len(data_q.shape) == 1:
                                data_q = np.expand_dims(data_q, axis=0)
                            if self.func == 'mean':
                                data_q = np.mean(data_q,axis=0)
                            elif self.func == 'max':
                                data_q = np.max(data_q,axis=0)
                        
                        elif current_fea == 'bert_pkl':
                            if len(data_q.shape) == 1:
                                data_q = np.expand_dims(data_q, axis=0)
                            if self.func == 'mean':
                                data_q = np.mean(data_q,axis=0)
                            elif self.func == 'max':
                                data_q = np.max(data_q,axis=0)
                        
                        elif current_fea == 'emobase_pkl':
                            data_q = data_q
                    
                        data = np.mean(np.array([data, data_q]),axis=0)
                y_data = self._Y[idx]
                output_data.append(data.astype(float))
                
            else:
                fea_output = []
                for path in self._X[idx]:
                    data = joblib.load(os.path.join(fea, path))
                    
                    current_fea = fea.split('/')[-2]
                    if current_fea == 'bhv_feature':    
                        data = np.array(data)      
                    elif current_fea == 'wav2vec2':
                        if len(data.shape) == 1:
                            data = np.expand_dims(data, axis=0)
                        if self.func == 'mean':
                            data = np.mean(data,axis=0)
                        elif self.func == 'max':
                            data = np.max(data,axis=0)
                    elif current_fea == 'bert_pkl':
                        if len(data.shape) == 1:
                            data = np.expand_dims(data, axis=0)
                        if self.func == 'mean':
                            data = np.mean(data,axis=0)
                        elif self.func == 'max':
                            data = np.max(data,axis=0)
                    elif current_fea == 'emobase_pkl':
                        data = data

                    # print(current_fea, data.shape)   
                    if self.conversation:
                        # import pdb;pdb.set_trace()
                        data_q = joblib.load(os.path.join(fea, path).replace('_a', '_q'))
                        if current_fea == 'bhv_feature':    
                            data_q = np.array(data_q)
                            data = np.hstack((data, data_q))
                        else:
                            if current_fea == 'wav2vec2':
                                if len(data_q.shape) == 1:
                                    data_q = np.expand_dims(data_q, axis=0)
                                if self.func == 'mean':
                                    data_q = np.mean(data_q,axis=0)
                                elif self.func == 'max':
                                    data_q = np.max(data_q,axis=0)
                            elif current_fea == 'bert_pkl':
                                if len(data_q.shape) == 1:
                                    data_q = np.expand_dims(data_q, axis=0)
                                if self.func == 'mean':
                                    data_q = np.mean(data_q,axis=0)
                                elif self.func == 'max':
                                    data_q = np.max(data_q,axis=0)
                            
                            elif current_fea == 'emobase_pkl':
                                data_q = data_q
                        
                            data = np.mean(np.array([data, data_q]),axis=0)
                    fea_output.append(data.astype(float))
                # import pdb;pdb.set_trace()
                
                same_len_array = np.zeros((self.wrap_len, fea_output[0].shape[0]))

                for ind, r in enumerate(fea_output):
                    same_len_array[ind,:] = r
                    
                output_data.append(same_len_array)
                y_data = np.array(self._Y[idx]).reshape(-1,1)
                # pdb.set_trace()
                # print(len(y_data), output_data[0].shape)
         
        return output_data, y_data    
    
    
class aac_speech_fusion_Dataset_analysis(Dataset):
    def __init__(self, ep_keys, eps_path, feature_root, wrap_len, conversation, func, label_weighted=False):
        
        self.wrap_len = wrap_len
        all_key = list(ep_keys.keys())
        if wrap_len != 1:
            output = []
            temp = []
            current_recording = all_key[0].split('/')[0]
            for i in range(len(all_key)):
                if len(temp) != wrap_len and current_recording == all_key[i].split('/')[0]:
                    temp.append(all_key[i])
                else:
                    if len(temp)==self.wrap_len:
                        output.append(temp.copy())
                    temp.clear()
                current_recording = all_key[i].split('/')[0]
                    
            self._X = output
            # pdb.set_trace()
            # random.shuffle(self._X)
            self._Y = []
            for i in self._X:
                self._Y.append([ep_keys[key] for key in i])
            flat_list = [1 if 1 in sublist else 0 for sublist in self._Y ]
            # pdb.set_trace()
        else:
            self._X = all_key
            random.shuffle(self._X)
            self._Y = [ep_keys[key] for key in self._X]
            flat_list = self._Y
        self.func = func
        self.feature_root = feature_root
        # pdb.set_trace()
        self.conversation = conversation
        assert len(self._X) == len(self._Y)
        # pdb.set_trace()
        if label_weighted:
            self.data_weight = []
            # self.label_weight = [1, 1]
            self.label_weight = [1/item[1] for item in sorted(Counter(flat_list).items())]
            for i in self._Y:
                if self.wrap_len==1:
                    if i == 0:
                        self.data_weight.append(self.label_weight[0])
                    elif i == 1:
                        self.data_weight.append(self.label_weight[1])
                else:
                    if 1 in i:
                        self.data_weight.append(self.label_weight[1])
                    else:
                        self.data_weight.append(self.label_weight[0])
        # pdb.set_trace()
                    
    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        output_data = []

        for fea in self.feature_root:
            
            if self.wrap_len == 1:
                data = joblib.load(os.path.join(fea, self._X[idx]))
                current_fea = fea.split('/')[-2]
                if current_fea == 'bhv_feature':    
                    data = np.array(data)
                    
                elif current_fea == 'wav2vec2':
                    if len(data.shape) == 1:
                        data = np.expand_dims(data, axis=0)
                    if self.func == 'mean':
                        data = np.mean(data,axis=0)
                    elif self.func == 'max':
                        data = np.max(data,axis=0)
                        
                elif current_fea == 'bert_pkl':
                    if len(data.shape) == 1:
                        data = np.expand_dims(data, axis=0)
                    if self.func == 'mean':
                        data = np.mean(data,axis=0)
                    elif self.func == 'max':
                        data = np.max(data,axis=0)
                
                elif current_fea == 'emobase_pkl':
                    data = data
                    
                
                if self.conversation:
                    data_q = joblib.load(os.path.join(fea, self._X[idx]).replace('_a', '_q'))
                    
                    if current_fea == 'bhv_feature':    
                        data_q = np.array(data_q)
                        data = np.hstack((data, data_q))
                    else:
                        if current_fea == 'wav2vec2':
                            if len(data_q.shape) == 1:
                                data_q = np.expand_dims(data_q, axis=0)
                            if self.func == 'mean':
                                data_q = np.mean(data_q,axis=0)
                            elif self.func == 'max':
                                data_q = np.max(data_q,axis=0)
                        
                        elif current_fea == 'bert_pkl':
                            if len(data_q.shape) == 1:
                                data_q = np.expand_dims(data_q, axis=0)
                            if self.func == 'mean':
                                data_q = np.mean(data_q,axis=0)
                            elif self.func == 'max':
                                data_q = np.max(data_q,axis=0)
                        
                        elif current_fea == 'emobase_pkl':
                            data_q = data_q
                    
                        data = np.mean(np.array([data, data_q]),axis=0)
                y_data = self._Y[idx]
                output_data.append(data.astype(float))
                
            else:
                fea_output = []
                for path in self._X[idx]:
                    data = joblib.load(os.path.join(fea, path))
                    
                    current_fea = fea.split('/')[-2]
                    if current_fea == 'bhv_feature':    
                        data = np.array(data)      
                    elif current_fea == 'wav2vec2':
                        if len(data.shape) == 1:
                            data = np.expand_dims(data, axis=0)
                        if self.func == 'mean':
                            data = np.mean(data,axis=0)
                        elif self.func == 'max':
                            data = np.max(data,axis=0)
                    elif current_fea == 'bert_pkl':
                        if len(data.shape) == 1:
                            data = np.expand_dims(data, axis=0)
                        if self.func == 'mean':
                            data = np.mean(data,axis=0)
                        elif self.func == 'max':
                            data = np.max(data,axis=0)
                    elif current_fea == 'emobase_pkl':
                        data = data

                    # print(current_fea, data.shape)   
                    if self.conversation:
                        data_q = joblib.load(os.path.join(fea, path).replace('_a', '_q'))
                        if current_fea == 'bhv_feature':    
                            data_q = np.array(data_q)
                            data = np.hstack((data, data_q))
                        else:
                            if current_fea == 'wav2vec2':
                                if len(data_q.shape) == 1:
                                    data_q = np.expand_dims(data_q, axis=0)
                                if self.func == 'mean':
                                    data_q = np.mean(data_q,axis=0)
                                elif self.func == 'max':
                                    data_q = np.max(data_q,axis=0)
                            elif current_fea == 'bert_pkl':
                                if len(data_q.shape) == 1:
                                    data_q = np.expand_dims(data_q, axis=0)
                                if self.func == 'mean':
                                    data_q = np.mean(data_q,axis=0)
                                elif self.func == 'max':
                                    data_q = np.max(data_q,axis=0)
                            
                            elif current_fea == 'emobase_pkl':
                                data_q = data_q
                        
                            data = np.mean(np.array([data, data_q]),axis=0)
                    fea_output.append(data.astype(float))
                # import pdb;pdb.set_trace()
                
                same_len_array = np.zeros((self.wrap_len, fea_output[0].shape[0]))

                for ind, r in enumerate(fea_output):
                    same_len_array[ind,:] = r
                    
                output_data.append(same_len_array)
                y_data = np.array(self._Y[idx]).reshape(-1,1)
                # pdb.set_trace()
                # print(len(y_data), output_data[0].shape)
        
        # pdb.set_trace()
        return self._X[idx], output_data, y_data    


#%%
if __name__=='__main__':
    fea_dict = {
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/171_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/172_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/173_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/174_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/175_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/176_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/177_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/178_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/179_a.pkl': 1,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/180_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/181_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/182_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/183_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/184_a.pkl': 1,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/185_a.pkl': 0,
    '/homes/GPU0/aac/speech_feature_extraction/wav2vec2/18/186_a.pkl': 0
    }
    dataset = aac_speech_bypath_Dataset(feature_dict = fea_dict, feature_name = 'wav2vec2', conversation=True, func = 'max', label_weighted=True)
    # if WEIGHTED_LABEL:
    weight = WeightedRandomSampler(dataset.data_weight, len(dataset))
    print(list(weight))
    # tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, sampler=weight, collate_fn=seq_collate_pad)
    train_loader = DataLoader(
        dataset,
        sampler=weight,
        batch_size=4
    )
    for x, y in train_loader:
        print(x.shape)
        print(y)