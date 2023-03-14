#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:29:28 2022

@author: shaohao
"""

import torch.nn as nn
import torch
import pdb

class DNN(nn.Module):
    def __init__(self, input_size, enc_size, Batch_norm=False):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.enc_sizes = [self.input_size]
        self.enc_sizes.extend(enc_size)

        linear_blocks = [nn.Linear(in_f, out_f, bias=True) for in_f, out_f
                         in zip(self.enc_sizes, self.enc_sizes[1:])]
        relu_blocks = [nn.ReLU() for i in range(len(linear_blocks)-1)]
        dropout_blocks = [nn.Dropout(0.25) for i in range(len(linear_blocks)-1)]
        if Batch_norm:
            batch_norm_blocks = [nn.BatchNorm1d(out_f) for _, out_f in
                                 zip(self.enc_sizes, self.enc_sizes[1:-1])]
            network = []
            for idx, block in enumerate(linear_blocks):
                network.append(block)
                if idx < len(linear_blocks)-1:
                    network.append(block)
                    network.append(batch_norm_blocks[idx])
        else:
            network = []
            for idx, block in enumerate(linear_blocks):
                network.append(block)
                if idx < len(linear_blocks)-1:
                    network.append(relu_blocks[idx])
                    network.append(dropout_blocks[idx])
        self.classifier = nn.Sequential(*network)

    def forward(self, x):
        prediction = self.classifier(x)
        # pdb.set_trace()

        return prediction
#%% tfm series
        
class self_Attn(nn.Module):
    def __init__(self, hidden_size, head):
        super(self_Attn, self).__init__()
        self.hidden_size = hidden_size
        self.head = head
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.head)

    def forward(self, encoder_outputs, src_len=None):
        pad_mask = torch.zeros((encoder_outputs.shape[1],encoder_outputs.shape[0])).type(torch.bool).to(encoder_outputs.device) # (B,T)
        if src_len != None:
            for idx, seq_len in enumerate(src_len):
                pad_mask[idx,seq_len:] = True
        attn_output, attn_output_weights = self.multihead_attn(query=encoder_outputs, key=encoder_outputs, value=encoder_outputs, key_padding_mask = pad_mask)
        return attn_output, attn_output_weights

class TFM(nn.Module):
    def __init__(self, layers, hidden_size, head):
        super(TFM, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.head = head
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.hidden_size, nhead=self.head, activation='gelu', dropout = 0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.layers, norm=nn.LayerNorm(self.hidden_size))
    
    def forward(self, encoder_outputs):
        out = self.transformer_encoder.forward(encoder_outputs)
        return out

class Trans_Encoder_clf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, hidden_layers_num, cl_num, tfm_head, max_length, dropout_r):
        super(Trans_Encoder_clf, self).__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers_num = hidden_layers_num
        self.cl_num = cl_num
        self.tfm_head = tfm_head
        # self.self_att_head = self_att_head
        self.max_length = max_length
        
        # input fc
        network = [nn.Linear(self.feat_dim, self.hidden_dim), nn.ReLU()]
        self.input_layer = nn.Sequential(*network)
        
        # tfm_encoder
        self.tfm = TFM(self.hidden_layers_num, self.hidden_dim, self.tfm_head)
        
        # drop_out
        self.drop_out = nn.Dropout(p=dropout_r)
        
        # output fc
        network = [nn.Linear(self.hidden_dim, self.cl_num), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, input_seqs, input_lens, hidden=None):

        input_fc_op = torch.zeros((len(input_seqs), self.max_length ,self.hidden_dim)).to(input_seqs.device)
        for i in range(len(input_lens)):
            input_fc_op[i, :input_lens[i], :] = self.input_layer.forward(input_seqs[i,:input_lens[i],:])
        
        # # tmf_emcoder
        input_fc_op = input_fc_op.transpose(0, 1)
        tfm_output = self.tfm.forward(input_fc_op)
        
        # # output fc
        tfm_output = tfm_output.transpose(0, 1)
        clf_output = []
        for i, data in enumerate(tfm_output):
            lens = input_lens[i]
            clf_output.append(self.clf.forward(data[:lens,:]))
            
        return tfm_output, clf_output

class multi_input_fusion_TFM(nn.Module):
    '''
    input_size_list: list with int input for each late fusion subnet
    enc_size_list: the subnet size (except the last element)
    '''
    def __init__(self, input_size_list, enc_size_list, max_length, device, Batch_norm=False):
        super(multi_input_fusion_TFM, self).__init__()
        self.Batchnorm = Batch_norm

        
        # entry dict
        self.sub_net_dict = nn.ModuleList()
        for i, input_size in enumerate(input_size_list):
            enc = [input_size]+enc_size_list[:-1]
            self.sub_net_dict.append(DNN(input_size, enc, self.Batchnorm).to(device))

        self.input_size = enc_size_list[-2]*len(self.sub_net_dict)
        # self.input_size = sum(input_size_list)
        self.enc_sizes = [self.input_size]
        self.enc_sizes.extend(enc_size_list)

        self.tfm = Trans_Encoder_clf(self.input_size, self.enc_sizes[-2], hidden_layers_num=2, cl_num=1, tfm_head=2, max_length = max_length, dropout_r=0.3).to(device)

    def forward(self, feature_in_list, input_lens, hidden=None):
        '''
        feature_in_list: list -> ele shape: (B,T,E)
        
        '''
        
        x = []
        for i, input_fea in enumerate(feature_in_list):
            # pdb.set_trace()
            x.append(self.sub_net_dict[i].forward(feature_in_list[i]))
        # pdb.set_trace()
        # x = torch.cat(feature_in_list, dim=2)
        x = torch.cat(x, dim=2)
        prediction = self.tfm(x, input_lens)
        # pdb.set_trace()
        # return prediction
        return prediction


#%%

class Trans_Encoder_clf_MIL(nn.Module):
    def __init__(self, feat_dim, hidden_dim, hidden_layers_num, cl_num, tfm_head, max_length, dropout_r, mil_mode):
        super(Trans_Encoder_clf_MIL, self).__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers_num = hidden_layers_num
        self.cl_num = cl_num
        self.tfm_head = tfm_head
        # self.self_att_head = self_att_head
        self.max_length = max_length
        self.mil_mode = mil_mode
        
        # input fc
        network = [nn.Linear(self.feat_dim, self.hidden_dim), nn.ReLU()]
        self.input_layer = nn.Sequential(*network)
        
        # tfm_encoder
        self.tfm = TFM(self.hidden_layers_num, self.hidden_dim, self.tfm_head)
        
        # drop_out
        self.drop_out = nn.Dropout(p=dropout_r)
        
        # output fc
        network = [nn.Linear(self.hidden_dim, self.cl_num), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
        self.mil_layer = MIL(self.max_length, self.mil_mode)
        
    def forward(self, input_seqs, input_lens, hidden=None):

        input_fc_op = torch.zeros((len(input_seqs), self.max_length ,self.hidden_dim)).to(input_seqs.device)
        for i in range(len(input_lens)):
            input_fc_op[i, :input_lens[i], :] = self.input_layer.forward(input_seqs[i,:input_lens[i],:])
        
        # # tmf_emcoder
        input_fc_op = input_fc_op.transpose(0, 1)
        tfm_output = self.tfm.forward(input_fc_op)
        
        # # output fc
        tfm_output = tfm_output.transpose(0, 1)
        clf_output = []
        for i, data in enumerate(tfm_output):
            lens = input_lens[i]
            clf_output.append(self.clf.forward(data[:lens,:]))
        clf_output = torch.cat(clf_output, dim =1).transpose(0,1)
        clf_output = self.mil_layer.forward(clf_output)
        
        return tfm_output, clf_output

class MIL(nn.Module):
    def __init__(self, time_step, mode):
        super(MIL, self).__init__()
        
        self.mode = mode
        self.time_step = time_step
        
        if self.mode in ['att_sigmoid','hybrid']:
            network = [nn.Linear(self.time_step, self.time_step), nn.Sigmoid()]
            self.mil_layer = nn.Sequential(*network)
        
    def forward(self, inputs):
        
        if self.mode == 'max':
            clf_output, _ = torch.max(inputs, dim=1)
            
        elif self.mode == 'average_pooling':
            clf_output = torch.mean(inputs, dim=1)
            
        elif self.mode == 'att_sigmoid':
            output_1 = self.mil_layer.forward(inputs)
            output_1 = output_1 * inputs
            clf_output = torch.mean(output_1, dim=1)
            
        elif self.mode == 'hybrid':
            # pdb.set_trace()
            output_1 = self.mil_layer.forward(inputs)
            output_1 = output_1 * inputs
            clf_output, _ = torch.max(output_1, dim=1)
        
        else:
            pdb.set_trace()
        
        return  clf_output   

class multi_input_fusion_TFM_MIL(nn.Module):
    '''
    input_size_list: list with int input for each late fusion subnet
    enc_size_list: the subnet size (except the last element)
    '''
    def __init__(self, input_size_list, enc_size_list, max_length, mode, device, Batch_norm=False):
        super(multi_input_fusion_TFM_MIL, self).__init__()
        self.Batchnorm = Batch_norm

        # entry dict
        self.sub_net_dict = nn.ModuleList()
        for i, input_size in enumerate(input_size_list):
            enc = [input_size]+enc_size_list[:-1]
            self.sub_net_dict.append(DNN(input_size, enc, self.Batchnorm).to(device))

        self.input_size = enc_size_list[-2]*len(self.sub_net_dict)
        # self.input_size = sum(input_size_list)
        self.enc_sizes = [self.input_size]
        self.enc_sizes.extend(enc_size_list)

        self.tfm = Trans_Encoder_clf_MIL(self.input_size, self.enc_sizes[-2], hidden_layers_num=2, cl_num=1, tfm_head=2, max_length = max_length, dropout_r=0.3, mil_mode = mode).to(device)

    def forward(self, feature_in_list, input_lens, hidden=None):
        '''
        feature_in_list: list -> ele shape: (B,T,E)
        
        '''
        
        x = []
        for i, input_fea in enumerate(feature_in_list):
            # pdb.set_trace()
            x.append(self.sub_net_dict[i].forward(feature_in_list[i]))
        x = torch.cat(x, dim=2)
        prediction = self.tfm(x, input_lens)
        # pdb.set_trace()
        # return prediction
        return prediction
#%%
# peter_mdf 20220901
class MIL_FBP(nn.Module): 
    def __init__(self, input_size, time_step, mode):
        super(MIL_FBP, self).__init__()
        
        self.mode = mode
        self.time_step = time_step
        self.input_size = input_size
        
        if self.mode == 'self_att':
            self.mil_layer = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=2,batch_first=True)
            self.sfm_f = nn.Softmax(dim=1)
        elif self.mode == 'att':
            network = [nn.Linear(self.input_size, self.time_step), nn.Sigmoid()]
            self.mil_layer = nn.Sequential(*network)
            
            
    def forward(self, inputs, based_feature):
        if self.mode == 'self_att':
            _, time_weight = self.mil_layer(based_feature, based_feature, based_feature)
            time_weight = time_weight.sum(dim=1)
            time_weight = self.sfm_f.forward(time_weight)
            clf_output = torch.sum(inputs*time_weight, dim=1)
            
        elif self.mode == 'att':
            # print('inputs', inputs.shape)
            output_1 = self.mil_layer.forward(based_feature)
            # print('output_1', output_1.shape)
            output_1 = torch.bmm(inputs.unsqueeze(1), output_1)
            clf_output = torch.mean(output_1, dim=1)
        
        else:
            pdb.set_trace()
        
        return  clf_output
    
class Trans_Encoder_clf_multitask_FBP(nn.Module):
    def __init__(self, feat_dim, hidden_dim, hidden_layers_num, cl_num, tfm_head, max_length, dropout_r):
        # peter_mdf 20220901
        super(Trans_Encoder_clf_multitask_FBP, self).__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers_num = hidden_layers_num
        self.cl_num = cl_num
        self.tfm_head = tfm_head
        # self.self_att_head = self_att_head
        self.max_length = max_length
        # self.mil_feature = mil_mode.split('_')[0]
        # self.mil_mode = mil_mode.split('_')[1]
        # input fc
        network = [nn.Linear(self.feat_dim, self.hidden_dim), nn.ReLU()]
        self.input_layer = nn.Sequential(*network)
        
        # tfm_encoder
        self.tfm = TFM(self.hidden_layers_num, self.hidden_dim, self.tfm_head)
        
        # drop_out
        self.drop_out = nn.Dropout(p=dropout_r)
        
        # output fc
        network = [nn.Linear(self.hidden_dim, self.cl_num), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
        # if self.mil_mode == 'Q' or self.mil_mode =='A':
        #     self.mil_layer = MIL_FBP(input_size, self.max_length, self.mil_mode)
        
    def forward(self, input_seqs, input_lens, hidden=None):

        input_fc_op = torch.zeros((len(input_seqs), self.max_length ,self.hidden_dim)).to(input_seqs.device)
        for i in range(len(input_lens)):
            input_fc_op[i, :input_lens[i], :] = self.input_layer.forward(input_seqs[i,:input_lens[i],:])
        
        # # tmf_emcoder
        input_fc_op = input_fc_op.transpose(0, 1)
        tfm_output = self.tfm.forward(input_fc_op)
        
        # # output fc
        tfm_output = tfm_output.transpose(0, 1)
        clf_output = []
        for i, data in enumerate(tfm_output):
            lens = input_lens[i]
            clf_output.append(self.clf.forward(data[:lens,:]))
        # pdb.set_trace()
        # mil_input = torch.cat(clf_output.copy(), dim =1).transpose(0,1)
        # print('mil_input', mil_input.shape)
        # mil_output = self.mil_layer.forward(mil_input)
            
        return clf_output

class multi_input_fusion_TFM_multitask_FBP(nn.Module):
    '''
    input_size_list: list with int input for each late fusion subnet
    enc_size_list: the subnet size (except the last element)
    '''
    def __init__(self, input_size_list, enc_size_list, max_length, mode, device, Batch_norm=False, fbp_size=64):
        super(multi_input_fusion_TFM_multitask_FBP, self).__init__()
        self.Batchnorm = Batch_norm
        self.mil_feature = mode.split('-')[0]
        self.mil_mode = mode.split('-')[1]        
        self.mil_based_fea_size = sum(input_size_list)//2
        self.fbp_size = fbp_size
        self.input_size_list = input_size_list
        # mil reduction dim layer
        self.fbp_input_layer = nn.Linear(self.mil_based_fea_size, self.fbp_size).to(device)
       
        # entry dict
        self.sub_net_dict = nn.ModuleList()
        for i, input_size in enumerate(input_size_list):
            enc = [input_size]+enc_size_list[:-1]
            self.sub_net_dict.append(DNN(input_size, enc, self.Batchnorm).to(device))

        self.input_size = enc_size_list[-2]*len(self.sub_net_dict)
        # self.input_size = sum(input_size_list)
        self.enc_sizes = [self.input_size]
        self.enc_sizes.extend(enc_size_list)

        self.tfm = Trans_Encoder_clf_multitask_FBP(self.input_size, self.enc_sizes[-2], hidden_layers_num=2, cl_num=1, tfm_head=2, max_length = max_length, dropout_r=0.3).to(device)
        self.mil_layer = MIL_FBP(self.fbp_size, max_length, self.mil_mode).to(device)
        
    def forward(self, feature_in_list, input_lens, hidden=None):
        '''
        feature_in_list: list -> ele shape: (B,T,E)
        
        '''
        
        # main path
        x = []
        mil_x = []
        for i, input_fea in enumerate(feature_in_list):
            # pdb.set_trace()
            x.append(self.sub_net_dict[i].forward(feature_in_list[i]))
            if self.mil_feature == 'Q':
                mil_x.append(feature_in_list[i][:,:,self.input_size_list[i]//2:])
            elif self.mil_feature == 'A':
                mil_x.append(feature_in_list[i][:,:,:self.input_size_list[i]//2])
            else:
                print('error')
                break
        mil_x =  torch.cat(mil_x, dim=2)
        mil_fbp_input = self.fbp_input_layer.forward(mil_x)
        # pdb.set_trace()
        # x = torch.cat(feature_in_list, dim=2)
        x = torch.cat(x, dim=2)
        prediction = self.tfm(x, input_lens)
        mil_input = torch.cat(prediction.copy(), dim = 1).transpose(0,1)
        prediction_mil = self.mil_layer.forward(mil_input, mil_fbp_input)
        # pdb.set_trace()
        # return prediction
        return prediction, prediction_mil
    
#%%  Bilstm multitask

class Bilstm_clf_multitask(nn.Module):
    def __init__(self, feature_dim, lstm_hidden_dim, input_hidden_dim, max_length, dropout_r, mil_mode):
        super(Bilstm_clf_multitask, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.input_hidden_dim = input_hidden_dim
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.dropout_r = dropout_r 
        lstm_0 = [nn.LSTM(input_hidden_dim, lstm_hidden_dim, num_layers=2, dropout=dropout_r, bidirectional=True)]
        self.lstm_model = nn.Sequential(*lstm_0)
        network = [nn.Linear(self.lstm_hidden_dim*2, 1), nn.Sigmoid()]
        self.output_layer = nn.Sequential(*network)
        network = [nn.Linear(self.feature_dim, self.input_hidden_dim)]
        self.input_layer = nn.Sequential(*network)
        
        self.tfm = TFM(2, self.lstm_hidden_dim*2, 2)
        # self.self_att = self_Attn(hidden_size=self.lstm_hidden_dim*2, head=8)
        
        self.h0 = torch.randn(2, 2, lstm_hidden_dim)
        self.c0 = torch.randn(2, 2, lstm_hidden_dim)
        
        self.mil_mode = mil_mode
        self.mil_layer = MIL(self.max_length, self.mil_mode)
        
    def forward(self, input_seqs, hidden=None):

        embedded = self.input_layer.forward(input_seqs) #[B,T,E]
        # pdb.set_trace()
        clf_output, (ht, ct) = self.lstm_model(embedded)
        
        tfm_input = clf_output.transpose(0, 1)
        
        attention_outputs = self.tfm(tfm_input)
        
        attention_outputs = attention_outputs.transpose(0, 1)
        
        op_layer_out = self.output_layer.forward(attention_outputs)
        
        mil_input = torch.clone(op_layer_out)
        mil_input = mil_input.squeeze(dim=2)
        mil_output = self.mil_layer.forward(mil_input)
        # pdb.set_trace()   
        
        return op_layer_out, mil_output

class multi_input_fusion_Bilstm_multitask(nn.Module):
    '''
    input_size_list: list with int input for each late fusion subnet
    enc_size_list: the subnet size (except the last element)
    '''
    def __init__(self, input_size_list, enc_size_list, max_length, mode, device, Batch_norm=False):
        super(multi_input_fusion_Bilstm_multitask, self).__init__()
        self.Batchnorm = Batch_norm
        
        # entry dict
        self.sub_net_dict = nn.ModuleList()
        for i, input_size in enumerate(input_size_list):
            enc = [input_size]+enc_size_list[:-1]
            self.sub_net_dict.append(DNN(input_size, enc, self.Batchnorm).to(device))

        self.input_size = enc_size_list[-2]*len(self.sub_net_dict)
        # self.input_size = sum(input_size_list)
        self.enc_sizes = [self.input_size]
        self.enc_sizes.extend(enc_size_list)
        
        self.lstm = Bilstm_clf_multitask(self.input_size, enc_size_list[-2], enc_size_list[-2], max_length = max_length, dropout_r=0.2, mil_mode=mode).to(device)

    def forward(self, feature_in_list, hidden=None):
        '''
        feature_in_list: list -> ele shape: (B,T,E)
        
        '''
        
        x = []
        for i, input_fea in enumerate(feature_in_list):
            # pdb.set_trace()
            x.append(self.sub_net_dict[i].forward(feature_in_list[i]))
        # pdb.set_trace()
        # x = torch.cat(feature_in_list, dim=2)
        x = torch.cat(x, dim=2)
        prediction, prediction_mil = self.lstm(x)
        # prediction_mil = prediction_mil.unsqueeze(dim=1)
        # pdb.set_trace()
        # return prediction
        return prediction, prediction_mil

#%%
class pretrain_link(nn.Module):
    '''
    input_size_list: list with int input for each late fusion subnet
    enc_size_list: the subnet size (except the last element)
    '''
    def __init__(self, mil_mode, max_length):
        super(pretrain_link, self).__init__()
        self.max_length = max_length
        self.mil_mode = mil_mode
        
        self.mil_layer = MIL(self.max_length, self.mil_mode)
        
    def forward(self, clf_output):
        '''
        feature_in_list: list -> ele shape: (B,T,E)
        
        '''
        # pdb.set_trace()
        origin_output = clf_output[1]
        mil_input = torch.cat(origin_output.copy(), dim =1).transpose(0,1)
        mil_output = self.mil_layer.forward(mil_input)
        
        return origin_output, mil_output
#%%
# if __name__=='__main__':
    
    # dnn feature
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # batch_size = 16
    # feature_size_list = [7,512,768]
    # enc_list = [512,128,128,3]
    # fea_list = [torch.rand(batch_size, feature_size).to(device) for feature_size in feature_size_list]
    # # dnn test
    # dnn_fusion = multi_input_fusion_DNN(feature_size_list, enc_list, device)
    # op = dnn_fusion.forward(fea_list)
    
    # # lstm feature
    # batch_size = 16
    # feature_size_list = [7,512,768]
    # enc_list = [512,128,128,3]
    # time_step = 5
    # fea_list = [torch.rand(batch_size, time_step, feature_size).to(device) for feature_size in feature_size_list]
    # seq_lengths = torch.LongTensor([time_step for i in range(batch_size)])
    # # lstm test
    # lstm_fusion = multi_input_fusion_Bilstm_multitask(feature_size_list, enc_list, time_step,  'max', device)
    # op, op2 = lstm_fusion.forward(fea_list, seq_lengths)

    # tfm feature
    # batch_size = 16
    # feature_size_list = [7*2, 768*2, 988*2]
    # enc_list = [128,128,2]
    # time_step = 15
    # fea_list = [torch.rand(batch_size, time_step, feature_size).to(device) for feature_size in feature_size_list]
    # seq_lengths = torch.LongTensor([time_step for i in range(batch_size)])
    # # tfm test
    # tfm_fusion = multi_input_fusion_TFM_multitask_FBP(feature_size_list, enc_list, time_step, 'Q-self_att', device)
    # op_indiv, op_mil = tfm_fusion.forward(fea_list, seq_lengths)