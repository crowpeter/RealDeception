#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:55:28 2022

@author: aac
"""

import pandas as pd
import glob
from pathlib import Path
import joblib
import os

INPUT_PATH = '/homes/GPU0/aac/new_label_33/'
OUT_PATH = '/homes/GPU0/aac/behavior_feature_extraction/bhv_feature/'
OUT_PATH_CSV = '/homes/GPU0/aac/behavior_feature_extraction/bhv_feature_csv/'
def time2sec(time):
    h,m,s = time.split(':')
    h=int(h)*3600
    m=int(m)*60
    return h+m+float(s)

# out_dict = {'video_idx':[],'q_a':[],'video_idx_q_a':[],\
#             'duration':[],'speed':[],'silence_duration':[],'silence_utterance_ratio':[],\
#             'duration_difference':[], 'duration_addition':[],'duration_ratio':[]}
for xlsx_path in glob.glob(INPUT_PATH+'/*.xlsx'):
    out_dict = {'video_idx':[],'q_a':[],'video_idx_q_a':[],\
                'duration':[],'speed':[],'silence_duration':[],'silence_utterance_ratio':[],\
                'duration_difference':[], 'duration_addition':[],'duration_ratio':[]}
    file_name = xlsx_path.split('/')[-1]
    
    data_clean = joblib.load('/homes/GPU0/aac/bert_aac/BERT_emb/BERT_emb_conversation/'+file_name.replace('.xlsx','.pkl'))
    ep = file_name.replace('(analysis).xlsx','')[1:]
    data = pd.read_excel(xlsx_path)

    # make path
    path = Path(OUT_PATH+'/'+ep)
    if not os.path.exists(str(path.as_posix())):
        path.mkdir(parents=True, exist_ok=True)    
    
    # last duration
    last_q_duration = 0
    last_a_duration = 0
    last_utt_end = 0
    if file_name == '010(analysis).xlsx':
        data = data[:138]
        del data['Unnamed: 0']
    
    for i in range(len(data)):
        # save_in_single = []
        video_id_flag = data.keys()[0]
        video_idx = data.iloc[i][video_id_flag]
        
        try:
            video_idx = int(video_idx)
            video_idx = str(int(data.iloc[i][video_id_flag]))
            q_a = 'q'
        except:
            video_idx = int(data.iloc[i-1][video_id_flag])
            video_idx = str(video_idx)
            q_a = 'a'
            
        st_flag = data.keys()[1]
        ed_flag = data.keys()[2]

        st_time = time2sec(str(data.iloc[i][st_flag]))
        ed_time = time2sec(str(data.iloc[i][ed_flag]))
          
        # compute duration
        current_d = ed_time-st_time

        # Speed
        speed = len(data_clean.iloc[i]['content'])/current_d

        # compute Silence-duration
        sil_d = ed_time - last_utt_end
        
        # compute Silence-utterance ratio
        sil_u_ratio = sil_d/current_d
        
        if q_a == 'a':
            # compute Duration difference
            d_diff = last_q_duration - current_d
            # compute Duration addition
            d_add = last_a_duration + current_d
            # compute Duration ratio
            if last_q_duration == 0:
                d_ratio = 0
            else:
                d_ratio = current_d/last_q_duration
            last_a_duration = current_d
            
        elif q_a == 'q':
            # compute Duration difference
            d_diff = last_a_duration - current_d
            # compute Duration addition
            d_add = last_q_duration + current_d
            # compute Duration ratio
            if last_a_duration == 0:
                d_ratio = 0
            else:
                d_ratio = current_d/last_a_duration
            last_q_duration = current_d
            
        last_utt_end = ed_time
        save_in_single = [current_d, speed, sil_d, sil_u_ratio, d_diff, d_add, d_ratio]
        joblib.dump(save_in_single, OUT_PATH+'/'+ep+'/'+video_idx+'_'+q_a+'.pkl')
        # save information in dict
        out_dict['video_idx'].append(video_idx)
        out_dict['q_a'].append(q_a)
        out_dict['video_idx_q_a'].append(video_idx+'_'+q_a)
        out_dict['duration'].append(current_d)
        out_dict['speed'].append(speed)
        out_dict['silence_duration'].append(sil_d)
        out_dict['silence_utterance_ratio'].append(sil_u_ratio)
        out_dict['duration_difference'].append(d_diff)
        out_dict['duration_addition'].append(d_add)
        out_dict['duration_ratio'].append(d_ratio)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(OUT_PATH_CSV+'/'+file_name.replace('.xlsx','.csv'))
    print(OUT_PATH_CSV+'/'+file_name.replace('.xlsx','.csv'), 'SAVE!!')