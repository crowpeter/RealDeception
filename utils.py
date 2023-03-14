#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:31:53 2022

@author: shaohao
"""

import glob
import joblib
import math
from os.path import exists
import random
import torch
#%%
class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler)//self.batch_size
        else:
            return (len(self.sampler) + self.batch_size -1)//self.batch_size
        
#%%
def check_load_fea(eps, fea_root_path, meta_path):
    
    '''
    check the feature loading
    
    input:
        eps: list -> eps index
        fea_root_path: list -> root of all feature you want to load, if multiple feature sets are selected, it will return the intersection of feature sets
    
    return:
        eps_feapath_label: dict: eps -> idx -> label
        non_dece_set: list -> truth eps
    '''
    
    non_dece_set = eps.copy()
    # load meta label
    eps_feapath_label = {}
    
    
    sta_0 = 0
    sta_1 = 1
    
    for path in glob.glob(meta_path+'/*.pkl'):
        ep_idx = path.split('/')[-1].replace('(analysis).pkl','')[1:]
        if ep_idx in eps:
            meta = joblib.load(path)
            # init eps block
            eps_feapath_label[ep_idx] = {}

            for i in range(len(meta)):
                idx = meta.iloc[i]['video_id']
                # check QA
                if not math.isnan(idx):
                    idx = str(int(idx))
                    
                    idx_flag = True
                    for sub_fea_root_path in fea_root_path:
                        # check there exist this "ep_idx + video_idx + q and a" in feature sets 
                        if exists(sub_fea_root_path+'/'+ep_idx+'/'+idx+'_q'+'.pkl') and \
                            exists(sub_fea_root_path+'/'+ep_idx+'/'+idx+'_a'+'.pkl'):
                                pass
                        else:
                            idx_flag = False
                            continue
                        
                        # check feature load
                        a_path = sub_fea_root_path+'/'+ep_idx+'/'+idx+'_a.pkl'
                        q_path = sub_fea_root_path+'/'+ep_idx+'/'+idx+'_q.pkl'
                        a_fea = joblib.load(a_path)
                        q_fea = joblib.load(q_path)                        

                        # check empty feature load
                        if len(a_fea) == 0 or len(q_fea) == 0:
                            # print(a_fea)
                            idx_flag = False
                            continue
                        
                    # all green
                    if idx_flag:
                        ori_label =  meta.iloc[i+1]['label']
                        try:
                            ori_label = int(ori_label)
                        except:
                            continue

                        # check label 
                        if ori_label == 0 or ori_label == 1:
                            eps_feapath_label[ep_idx][idx] = 0
                            sta_0+=1
                        elif ori_label == 2:
                            eps_feapath_label[ep_idx][idx] = 1
                            sta_1+=1
                            # remove lie eps
                            if ep_idx in non_dece_set:
                                non_dece_set.remove(ep_idx)
                        else:
                            continue
    print('total truth sample:', sta_0)       
    print('total deception sample:', sta_1) 
    return eps_feapath_label, non_dece_set

#%%
def manual_fold_cv(fold_num, eps, non_dece_set):
    # manual fold
    fold=fold_num
    cv_len_variable = (len(eps)//fold)+1
    # ts_list = np.random.permutation(cv_len_variable)
    truth_glob_box = non_dece_set.copy()
    lie_glob_box = list(set(eps)-set(non_dece_set))
    # [[[train_ep_idx],[test_ep_idx]]]
    cv_ts_box = []
    
    truth_used_box = []
    lie_used_box = []
    for cv_idx in range(fold):
        # print(cv_idx)
        tr_ts = [[],[]]
    
        lie_box = list(set(lie_glob_box) - set(lie_used_box))
        truth_box = list(set(truth_glob_box) - set(truth_used_box))
        
        if cv_idx < 4:
            lie_ts_ele = random.sample(lie_box, cv_len_variable//2)
            truth_ts_ele = random.sample(truth_box, cv_len_variable - len(lie_ts_ele))
        else:
            lie_ts_ele = lie_box
            truth_ts_ele = truth_box
            
        lie_used_box.extend(lie_ts_ele)
        truth_used_box.extend(truth_ts_ele)
        
        tr_ts[0] = lie_ts_ele
        tr_ts[0].extend(truth_ts_ele)
        
        tr_ts[1] = list(set(lie_glob_box)-set(lie_ts_ele))
        tr_ts[1].extend(list(set(truth_glob_box)-set(truth_ts_ele)))
        cv_ts_box.append(tr_ts)
    return cv_ts_box
# pdb.set_trace()
#%%

def collate_fn(batch):
    batch = list(zip(*batch))
    # import pdb;pdb.set_trace()
    labels = torch.tensor(batch[1], dtype=torch.int32)
    texts = batch[0]
    del batch
    return texts, labels




