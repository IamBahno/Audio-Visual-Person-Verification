# --------------------------------------------------------
# MultiHead-Factorized-Attentive-Pooling
# Github source: https://github.com/BUTSpeechFIT/Wespeaker_SSL/blob/f1b5631fa4e8ce04bcb2777829371fa22ba86927/wespeaker/models/Transformer_WavLM_Large.py
# Copyright (c) 2024 Peng Junyi
# Licensed under Apache License 2.0
# --------------------------------------------------------

# i added method get_emb_and_lower_levels() for retrieving lower level representations from embedding extractor

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from newWavlm import *
from einops import rearrange, repeat
from torch.nn.utils import remove_weight_norm
from new_modules import GradMultiply
# from wespeaker.models.ssl_backend import *


class MHFA(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k) # B, T, H

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2) # B, T, 1

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs

class WavLM_Base_MHFA(nn.Module):
    def __init__(self,model_path, pooling, head_nb, embed_dim, group,cnn_scale=0.0,layer_drop=0.05):
        super(WavLM_Base_MHFA, self).__init__()


        cfg = WavLMConfig()
        self.model = WavLM(cfg)
        if pooling == 'MHFA':
            self.back_end = MHFA(head_nb=head_nb,outputs_dim=embed_dim)
            


    def forward(self,wav_and_flag):
        
        x = wav_and_flag
        # with torch.no_grad():
        rep, layer_results = self.model.extract_features(x[:,:16000*20], output_layer=13)
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)
        
        spk_embedding = self.back_end(x)
        return spk_embedding

    def get_emb_and_lower_levels(self,wav_and_flag):
        
        x = wav_and_flag
        # with torch.no_grad():
        rep, layer_results = self.model.extract_features(x[:,:16000*20], output_layer=13)
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)
        inner1 = x[:,:,:,1]
        inner2 = x[:,:,:,3]
        inner3 = x[:,:,:,5]
        inner1 = torch.mean(inner1,dim=2)
        inner2= torch.mean(inner2,dim=2)
        inner3= torch.mean(inner3,dim=2)
        spk_embedding = self.back_end(x)
        return spk_embedding,[torch.nn.functional.normalize(inner1),torch.nn.functional.normalize(inner2),torch.nn.functional.normalize(inner3)]
