#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:35:26 2022

@author: shaohao
"""

import os
import argparse
import subprocess 
#import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from glob import glob
import csv
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

param_grid = {
    'feature': [
                  'bert_pkl+emobase_pkl+bhv_feature',
                ],
    'conversation': [
                    'yes', 
                    ],
    'functional': ['mean'],
    'model':['TRANS'],
    'learning_rate': [1e-4],
    'wrap': [15],
    'alpha': [0.5],
    'mil_mode': ['max', 'average_pooling', 'att_sigmoid', 'hybrid'],
}
python_path=f'./run_multitask.py'

dir_path=os.path.dirname(python_path)
python_path=os.path.basename(python_path)
for idx,param in tqdm(enumerate(ParameterGrid(param_grid))):
    #if idx <80: continue
    # print(idx, param)
    arg_now=' '.join(['--{} {} '.format(key,v) for key,v in param.items()])
    exec_line=f'python {python_path} {arg_now}'
    print ('--',idx,'-- ',exec_line)
    subprocess.call(exec_line,shell=True)
