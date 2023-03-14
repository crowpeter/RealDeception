#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:27:49 2022

@author: aac
"""

from glob import glob
import os
from argparse import ArgumentParser
import subprocess as sp
from os.path import exists

def subprocess_cmd(cmd):
    p = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
    # Grab stdout line by line as it becomes available.  This will loop until
    # p terminates.
    while p.poll() is None:
        print(p.stdout.readline())


parser = ArgumentParser(description='Extract feature from opensmile')
parser.add_argument('--Extract_bin_PATH', type=str,
                    default='/homes/GPU0/aac/opensmile/build/progsrc/smilextract/SMILExtract')
parser.add_argument('--Conf_PATH', type=str,
                    default='/homes/GPU0/aac//opensmile/config/emobase/emobase.conf')
parser.add_argument('--DATA_DIR', type=str,
                    default='/homes/GPU0/aac/wav2vec_aac/cut_wav_33_16k/')
parser.add_argument('--OUT_DIR', type=str,
                    default='/homes/GPU0/aac/speech_feature_extraction/emobase/')
args = parser.parse_args()

Extract_bin_PATH = args.Extract_bin_PATH
Conf_PATH = args.Conf_PATH
DATA_DIR = args.DATA_DIR
OUT_DIR = args.OUT_DIR
# %%
# ==============================================================================
# NNIME
# ==============================================================================


def command(
    i, o): return '{} -C {} -I {} -O {}'.format(Extract_bin_PATH, Conf_PATH, i, o)


original_dir = DATA_DIR
output_dir = OUT_DIR
# ==============================================================================
# OpenSmile bat
# ifiles = glob(DATA_DIR + '/**/*.wav', recursive=True)
with open('error_list_33.txt') as f:
    err_list = f.readlines()
err_list = [err[:-1] for err in err_list]

ifiles = []
for wav_path in glob(DATA_DIR + '/**/*.wav', recursive=True):
    ep_idx = wav_path.split('/')[-2]
    utt_idx =  wav_path.split('/')[-1].replace('.wav','')
    a_q = utt_idx.split('_')[-1]
    
    if wav_path in err_list:
        continue    
    # error
    # if ep_idx == '11' and utt_idx == '623_a':
    #     continue
    # elif ep_idx == '11' and utt_idx == '623_q':
    #     continue
    
    # elif ep_idx == '13' and utt_idx == '377_a':
    #     continue
    # elif ep_idx == '13' and utt_idx == '377_q':
    #     continue
    
    # elif a_q == 'q':
    #     if exists(DATA_DIR+ep_idx+'/'+utt_idx[:-1]+'a'+'.wav'):
    #         pass
    #     else:
    #         continue
    # elif a_q == 'a':
    #     if exists(DATA_DIR+ep_idx+'/'+utt_idx[:-1]+'q'+'.wav'):
    #         pass
    #     else:
    #         continue
    ifiles.append(wav_path)
    
ofiles = map(lambda x: x.replace(
    original_dir, output_dir).replace('.wav', '.arff'), ifiles)
mkdir_ofiles = list(map(lambda x: x.replace(
    original_dir, output_dir).replace('.wav', '.arff'), ifiles))
for out_dir in mkdir_ofiles:
    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))
cmd_line = map(command, ifiles, ofiles)

########
#-SAVE-#
########
sfile = OUT_DIR + \
    '{}.sh'.format(os.path.basename(Conf_PATH).replace('.conf', ''))
with open(sfile, 'w') as s:
    s.write('\n'.join(cmd_line))

# subprocess_cmd('./{}'.format(sfile))
