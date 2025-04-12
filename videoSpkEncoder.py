# --------------------------------------------------------
# MultiHead-Factorized-Attentive-Pooling
# Github source: https://github.com/BUTSpeechFIT/Wespeaker_SSL/blob/f1b5631fa4e8ce04bcb2777829371fa22ba86927/wespeaker/models/Transformer_WavLM_Large.py
# Copyright (c) 2024 Peng Junyi
# Licensed under Apache License 2.0
# --------------------------------------------------------

# i added classes WavLM_Base_MHFA_fusion() MHFA_fusion() 
#         for frame-wise cross modality soft attention
# and upsample function


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
from torch.nn.functional import normalize 


# upsample face embeddings 
def upsample(image_embeddings,to_n_samples):
    num_samples = image_embeddings.shape[1]

    # Define the desired number of samples after upsampling
    desired_num_samples = to_n_samples

    # Calculate the base number of repetitions for each sample
    base_repetitions = desired_num_samples // num_samples

    # Calculate the remaining number of frames after filling with base repetitions
    remaining_frames = desired_num_samples % num_samples

    # Randomly distribute the remaining frames across the samples
    repetitions = torch.ones(num_samples, dtype=torch.long,device=torch.device('cuda', torch.cuda.current_device())) * base_repetitions

    if remaining_frames > 0:
        indices = torch.randperm(num_samples)[:remaining_frames]
        repetitions[indices] += 1

    # Repeat each sample by the calculated repetitions
    upsampled_tensor = image_embeddings.repeat_interleave(repetitions, dim=1)

    return upsampled_tensor

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


# implementing the frame-wise cross modality soft attention (mhfa,image emmbeddings tranformations and fusion)
class MHFA_fusion(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA_fusion, self).__init__()

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


        #fusion part
        self.weights_quality = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.attention_layer = nn.Linear(512+768,2)
        self.image_transform = nn.Linear(512,512)
        self.audio_transform = nn.Linear(256,512)
        self.last_image_transform = nn.Linear(512,512)



    def forward(self, x,image_embeddings):
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


        #video part ################################

        quality = torch.sum(x.mul(nn.functional.softmax(self.weights_quality, dim=-1)), dim=-1).transpose(1, 2)
        image_embeddings = upsample(image_embeddings,k.shape[1])
        quality = normalize(quality,dim=2)
        image_embeddings = normalize(image_embeddings,dim=2)
        embs_for_quality = torch.cat((quality,image_embeddings),dim=2)
        attentions = self.attention_layer(embs_for_quality)  #count attention for all frames
        attentions = nn.functional.softmax(attentions,dim=2) #conver them to real weigths
        v = v * attentions[:,:,0].unsqueeze(-1) #multiply audio values with its weights
        image_embeddings = self.image_transform(image_embeddings) #transform image
        image_embeddings = image_embeddings * attentions[:,:,1].unsqueeze(-1) #multiplu images with its weights
        mean_image_embeddings = torch.mean(image_embeddings,dim=1)  # mean between image frames

        #video part ################################


        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2) # B, T, 1

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        # transform audio embeddings from 256 to 512 so it matches#################################
        outs = normalize(outs)
        outs = self.audio_transform(outs)
        mean_image_embeddings = normalize(mean_image_embeddings)
        mean_image_embeddings = self.last_image_transform(mean_image_embeddings)
        outs = outs + mean_image_embeddings
        #########################################################################

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

# class for retrieving hidden units from audio transformer
# and and calling the fusion model
class WavLM_Base_MHFA_fusion(nn.Module):
    def __init__(self,model_path, pooling, head_nb, embed_dim, group,cnn_scale=0.0,layer_drop=0.05,joint_training = False):
        super(WavLM_Base_MHFA_fusion, self).__init__()


        cfg = WavLMConfig()
        self.model = WavLM(cfg)
        if pooling == 'MHFA':
            self.back_end = MHFA_fusion(head_nb=head_nb,outputs_dim=embed_dim)
        self.joint_training = joint_training


    def forward(self,wav_and_flag,images):
        
        x = wav_and_flag
        if(self.joint_training == False):
            with torch.no_grad():
                rep, layer_results = self.model.extract_features(x[:,:16000*20], output_layer=13)
        else:
            rep, layer_results = self.model.extract_features(x[:,:16000*20], output_layer=13)
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)
        
        spk_embedding = self.back_end(x,images)
        return spk_embedding
