#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:08:53 2022

@author: aac
"""


from os.path import basename
from glob import glob
from collections import defaultdict
import joblib as ib
import os
import numpy as np
from argparse import ArgumentParser
import pathlib
def read_arff(feat_path, feat_type):
    if feat_type == 'ComParE' or feat_type == 'EGEMAPS' or feat_type == 'emo_large' or feat_type == 'Emobase':
        with open(feat_path) as f:
            content_tmp = f.readlines()
        content = content_tmp[-1].split(',')
        feat = np.array([float(content[i+1]) for i in range(len(content)-2)])
    elif feat_type == 'ComParE_LLD':
        with open(feat_path) as f:
            content_tmp = f.readlines()
        content = content_tmp[138:]
        content_1 = [content[i].split(',') for i in range(len(content))]
        frame_num = len(content_1)
        feat = np.zeros((frame_num, 131))
        for j in range(frame_num):
            feat[j,:] = np.array([float(content_1[j][k+1]) for k in range(len(content_1[j])-2)])
    elif feat_type == 'EGEMAPS_LLD':
        with open(feat_path) as f:
            content_tmp = f.readlines()
        content = content_tmp[31:]
        content_1 = [content[i].split(',') for i in range(len(content))]
        frame_num = len(content_1)
        feat = np.zeros((frame_num, 24))
        for j in range(frame_num):
            feat[j,:] = np.array([float(content_1[j][k+1]) for k in range(len(content_1[j])-2)])
    elif feat_type == 'Emobase_LLD':
        with open(feat_path) as f:
            content_tmp = f.readlines()
        content = content_tmp[86:]
        content_1 = [content[i].split(',') for i in range(len(content))]
        frame_num = len(content_1)
        feat = np.zeros((frame_num, 78))
        for j in range(frame_num):
            feat[j,:] = np.array([float(content_1[j][k+2]) for k in range(len(content_1[j])-3)])
    return feat


parser = ArgumentParser(description='Save feature from arff file')
parser.add_argument('--FEAT_DIR', type=str, default='./emobase/', help='Put your arff main folder here')
parser.add_argument('--SAVE_DIR', type=str, default='./emobase/', help='dir you want to save the pkl file')
parser.add_argument('--Conf', type=str, default='Emobase', help='Only Support ComParE(_LLD), EGEMAPS(_LLD), emo_large, Emobase(_LLD)')

args = parser.parse_args()
#%%
Conf = args.Conf
FEAT_DIR = args.FEAT_DIR
SAVE_DIR = args.SAVE_DIR

# All_feat = defaultdict()
All_arff_files = glob(FEAT_DIR + '/**/*.arff', recursive=True)
for files in All_arff_files:
    feat = read_arff(files, Conf)
    # key = basename(files).replace('.arff', '')
    files = files.replace('emobase', 'emobase_pkl').replace('.arff','.pkl')
    p = pathlib.Path('/'.join(files.split('/')[:-1]))
    p.mkdir(parents=True, exist_ok=True)
    ib.dump(feat, files)
    # All_feat[key] = feat

# ib.dump(SAVE_DIR+'/All_feat.pkl')
