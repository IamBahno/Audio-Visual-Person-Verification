#the main training script for models working on videos
# in the main function: set paths to models and datasets
#                      choose if data should be augmented
#                      choose if to joint train with the embeddings extractors
#                      choose if to use the frame-wise cross modality attention
# if not using frame-wise cross modality attention choose the fusion model in class AudioVisual(nn.Module)
# before running create folder "training0" fot logging and saving models
#                 in DatasetLoader: set number of identities and samples in dataset
#                                    and paths to pickles with paths of samples

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import videoSpkEncoder
import torch.optim as optim
from torch.utils.data import DataLoader
from videoDatasetLoader import CustomDataset
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
import logging
import os
#ddp
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce,reduce,ReduceOp
import time
from inception_resnet_v1 import InceptionResnetV1
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import torchaudio
import torchvision
from torch.nn.functional import normalize 
from aam_softmax import ArcFace
from torchvision.transforms.v2 import RandomChoice
import numpy as np
from sklearn.metrics import det_curve

#load in batch of input data and augment it for evaluation
def data_augmented_image_and_wav_loader(paths,audio_dataset_path,video_dataset_path,musan_path,musan_dict):
    # vox_1_test_untarred\id10298\f5eaTNf7-io\00002.wav
    images = []
    audios = []
    ##############    AUGMENT PART 1       ##########################
    blur = transforms.GaussianBlur(7,(0.1,3))
    crop = transforms.Compose([
        transforms.RandomCrop(size=(70,70)),
        transforms.Resize(size=(112,112))
    ])
    add_noice = torchaudio.transforms.AddNoise()
    rand_transform = RandomChoice([blur,crop])
    ################################################################
    for path in paths:
        if(path[-len("00010.wav"):] == "00010.wav"):
            path = path[:-len("00010.wav")] + "000010.wav"
        try:
            image = torchvision.io.read_image(os.path.join(video_dataset_path,f"{path[:-4]}.jpg")).to(0)
        except:
            audios.append(torch.zeros((48000)).to(0))
            images.append(torch.zeros((75, 3, 112, 112)).to(0))
            continue
        image = image / 255
        image = torch.split(image, 112, dim=2)
        image = torch.stack(image, dim=0)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        try:
            speech, sample_rate = torchaudio.load(os.path.join(audio_dataset_path,path))
            speech = speech.to(0)
        except:
            print(os.path.join(audio_dataset_path,path))
            print("audio load fail")
            continue
        speech = speech[0][ (len(speech[0]) // 2) - 24_000: (len(speech[0]) // 2) + 24_000].to(0)


        # DATA AUGMENTAION  2 #################################################
        if(torch.rand(1) < 0.8): # augment in 80%
            #apply image augmentation
            if(torch.rand(1) < 0.5):
                start = random.randint(0, 75 - 1)
                length = random.randint(1, 75 - start)
                subsequence = image[start:start + length]
                subsequence = rand_transform(subsequence)
                image[start:start+length] = subsequence
            else:#apply audio augmentation
                if(torch.rand(1) < 0.5): # add musan
                    wav_input, sample_rate = torchaudio.load(os.path.join(musan_path, musan_dict[int(torch.randint(low=0,high=len(musan_dict),size=(1,)))])) #load a tensor
                    wav_input.to(0)
                    if(wav_input.size(1) >= 48_000):
                        noice =wav_input[0,(wav_input.size(1) // 2)- 24_000 : (wav_input.size(1) // 2)+24_000].to(0)
                    else:
                        pad_amount = 48_000 - wav_input.size(1)
                        noice = torch.nn.functional.pad(wav_input[0], (0,pad_amount),value=0).to(0)
                else: # add random noice
                    noice = torch.rand_like(speech)
                    noice.to(0)
                    noice = (noice * 2 )- 1
                speech = add_noice(speech,noice,torch.tensor(1).to(0))
        # #################################################



        img=transform(image)
        audios.append(speech)
        images.append(img)
    images = torch.stack(images).to(0)
    audios = torch.stack(audios).to(0)
    return images,audios

#load in batch of input data for evaluation
def image_and_wav_loader(paths,audio_dataset_path,video_dataset_path):
    # vox_1_test_untarred\id10298\f5eaTNf7-io\00002.wav
    images = []
    audios = []

    for path in paths:
        if(path[-len("00010.wav"):] == "00010.wav"): #fix error in names of saved wavs
            path = path[:-len("00010.wav")] + "000010.wav"
        try:
            img = torchvision.io.read_image(os.path.join(video_dataset_path,f"{path[:-4]}.jpg")).to(0)
        except:
            audios.append(torch.zeros((48000)).to(0))
            images.append(torch.zeros((75, 3, 112, 112)).to(0))
            continue
        img = img / 255
        img = torch.split(img, 112, dim=2)
        img = torch.stack(img, dim=0)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        img=transform(img)
        images.append(img)
        try:
            wav_input, sample_rate = torchaudio.load(os.path.join(audio_dataset_path,path))
            wav_input = wav_input.to(0)
        except:
            print(os.path.join(audio_dataset_path,path))
            print("audio load fail")
            continue
        wav_input = wav_input[0][ (len(wav_input[0]) // 2) - 24_000: (len(wav_input[0]) // 2) + 24_000].to(0)
        audios.append(wav_input)
    images = torch.stack(images).to(0)
    audios = torch.stack(audios).to(0)
    return images,audios

#get embedding for evaluation
def get_audio_vis_embs(images,audios,model):
    with torch.no_grad():
        model.eval()
        try:
            embs = model(audios,images)
        except:
            return None

    return embs

# function borrowed from: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
# Copyright (c) 2020-present NAVER Corp.
# Licensed under The MIT License 
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

#evaluate given model on verification trial, just to see the results
def verificate(model,data_augmentation):
    model = model.module.audio_vis_model
    audio_dataset_path = "/flash/project_465000792/xbahou00/vox_1_test"
    video_dataset_path = "/flash/project_465000792/xbahou00/vox_1_test"
    trials_path = "voxceleb1pairs.ssv"


    #data augment #############################################
    musan_path = "/flash/project_465000792/xbahou00/musan"
    musan_dict = {}
    if(data_augmentation == True):
        wav_paths = []
        for name in os.listdir(musan_path):
            if (name in ["music","noise"]):
                for dirpath, dirnames, filenames in os.walk(os.path.join(musan_path,name)):
                    for filename in filenames:
                        # Check if the file has a .wav extension
                        if filename.endswith('.wav'):
                            # Construct the full path to the .wav file
                            wav_path = os.path.join(dirpath, filename)
                            # Append the path to the list
                            wav_paths.append(wav_path[len(musan_path)+1:])

        for i,path in enumerate(wav_paths):
            musan_dict[i] = path
    ##############################################
    with open(trials_path, "r") as file:
        # Iterate over each line in the file
        all_paths = []
        labels = []
        paths1 = []
        paths2 = []
        for line in file:
            # Split the line into parts based on whitespace
            parts = line.strip().split()

            # Extract the label (0 or 1) and file paths
            label = int(parts[0])
            file_path1 = parts[1]
            file_path2 = parts[2]

            labels.append(label)
            paths1.append(file_path1)
            paths2.append(file_path2)
            all_paths.append(file_path1)
            all_paths.append(file_path2)
        all_paths = set(all_paths)

    all_paths = list(all_paths)
    all_paths.sort()

    batch_size = 50
    path2emb = {}
    for i in range(0,len(all_paths),batch_size):
        j = i + batch_size if ((i + batch_size) <len(all_paths) ) else len(all_paths)
        if(data_augmentation == False):
            output = image_and_wav_loader(all_paths[i:j],audio_dataset_path,video_dataset_path)
        else:
            output = data_augmented_image_and_wav_loader(all_paths[i:j],audio_dataset_path,video_dataset_path,musan_path,musan_dict)

        images,audios = output

        ######################################################
        #           FUSION
        embs = get_audio_vis_embs(images,audios,model)

        if(embs == None):
            print("None")
            continue
        ######################################################

        # #################################################
        # ##     CONC
        # with torch.no_grad():
        #     emb1 = audio_model(audios)
        #     emb2 = face_model(images)

        # if(emb1 == None or emb2 == None):
        #     continue
        
        # emb1 = normalize(emb1)
        # emb2 = normalize(emb2)

        # embs = torch.cat((emb1,emb2),dim=1)
        # #################################################

        for i,path in enumerate(all_paths[i:j]):
            if( torch.all(audios[i] == 0)):
                path2emb[path] = None
            else:
                path2emb[path] = embs[i]

    old_real_labels = []
    old_embs1 = []
    old_embs2 = []

    for (label,path1,path2) in zip(labels,paths1,paths2):
        try:
            emb1 = path2emb[path1]
            emb2 = path2emb[path2]
        except:
            continue
        if(emb1 == None or emb2 == None):
            continue
        
        old_real_labels.append(label)
        old_embs1.append(emb1)
        old_embs2.append(emb2)

    real_labels = torch.tensor(old_real_labels)
    embs1 = torch.stack(old_embs1)
    embs2 = torch.stack(old_embs2)

    # if it is 1:1 embedding
    if(embs1.dim() == 2 ):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_similarities = cos(embs1, embs2)
    # if it is N:M embedding
    else:
        cos_similarities = []
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for person1,person2 in zip(embs1,embs2):
            best_sim = - 1
            for emb1 in person1:
                for emb2 in person2:
                    sim = cos(emb1,emb2)
                    if(sim > best_sim):
                        best_sim = sim
            cos_similarities.append(best_sim)
        cos_similarities = torch.stack(cos_similarities)

    # fpr, fnr, thresholds = det_curve(real_labels, cos_similarities)
    # eer1 = compute_eer(fpr,fnr)
    # print(f"EER1: {eer1*100} %")


    #funkci eer z stack overflow
    cos_similarities = cos_similarities.cpu()
    fpr, tpr, thresholds = roc_curve(real_labels, cos_similarities)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    with open("training0/training_log.txt","a") as f:
        f.write(f"EER: {eer * 100}%\n")
    fpr, fnr, thresholds = det_curve(real_labels, cos_similarities)
    min_dcf,min_dcf_thresh = ComputeMinDcf(fnr,fpr,thresholds,0.01,1,1)
    with open("training0/training_log.txt","a") as f:
        f.write(f"min dcf: {min_dcf}")
    return eer * 100

#init procces group for distributed training
def ddp_setup():
    #uncomment the commented part if not using torchrun

    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "7888"
    # os.environ["MASTER_PORT"] = "7890"

    init_process_group(backend="gloo")
    # init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

#check if all the inputs are zeros, meaning its dummy input due to some data error
def check_dummy(audio_input,video_input,labels):
    for i,label in enumerate(labels):
        if torch.count_nonzero(label) == 0:
            audio_input = torch.cat((audio_input[:i], audio_input[i + 1:]), dim=0)
            video_input = torch.cat((video_input[:i], video_input[i + 1:]), dim=0)
            labels = torch.cat((labels[:i], labels[i + 1:]), dim=0)
            break
    for i,label in enumerate(labels):
        if torch.count_nonzero(label) == 0:
            audio_input = torch.cat((audio_input[:i], audio_input[i + 1:]), dim=0)
            video_input = torch.cat((video_input[:i], video_input[i + 1:]), dim=0)
            labels = torch.cat((labels[:i], labels[i + 1:]), dim=0)
            break
    for i,label in enumerate(labels):
        if torch.count_nonzero(label) == 0:
            audio_input = torch.cat((audio_input[:i], audio_input[i + 1:]), dim=0)
            video_input = torch.cat((video_input[:i], video_input[i + 1:]), dim=0)
            labels = torch.cat((labels[:i], labels[i + 1:]), dim=0)
            break
    return audio_input,video_input,labels

# implementation of softmax cross entropy loss layer
class SoftMaxCrossEntropyLoss(nn.Module):
    def __init__(self,input_dimension,number_of_classes):
        super(SoftMaxCrossEntropyLoss, self).__init__()

        self.lin_layer = nn.Linear(input_dimension,number_of_classes,bias=False)
        torch.nn.init.xavier_normal_(self.lin_layer.weight)

        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self,inputs,labels):
        predictions = self.lin_layer(inputs)

        loss = self.loss_func(predictions,labels)
        return loss

#mean across all face embeddings concatenated with audio embedding 
class ConcatenationFusion(nn.Module):
    def __init__(self,audio_dim,image_dim,out_dim):
        super(ConcatenationFusion, self).__init__()

    def forward(self,audio_input,video_input):
        video_input = torch.mean(video_input,dim=1)
        audio_input = normalize(audio_input)
        video_input = normalize(video_input)
        out = torch.cat((audio_input,video_input),dim=1)
        return out


#mean across all face embeddings + multimodal soft attention
class MeanSoftAttentionFusion(nn.Module):
    def __init__(self,audio_dim,image_dim,out_dim):
        super(MeanSoftAttentionFusion, self).__init__()

        #fusion part
        self.audio_transform = nn.Linear(audio_dim, out_dim)
        self.image_transform= nn.Linear(image_dim, out_dim)

        self.cross_modality_attention = nn.Linear(image_dim+audio_dim,2)

    def forward(self,audio_input,video_input):
        video_input = torch.mean(video_input,dim=1)

        audio_input = normalize(audio_input)
        video_input = normalize(video_input)


        multi_mod_attention = self.cross_modality_attention(torch.cat((audio_input,video_input),dim=1))
        multi_mod_attention = nn.functional.softmax(multi_mod_attention, dim=1)
        
        audio_emb_transformed = self.audio_transform(audio_input)
        face_emb_transformed = self.image_transform(video_input)


        weighted_audio_emb = audio_emb_transformed * multi_mod_attention[:,0].unsqueeze(1)
        weighted_face_emb = face_emb_transformed * multi_mod_attention[:,1].unsqueeze(1)
        emb = torch.add(weighted_audio_emb,weighted_face_emb)
        return emb

# attention for video frames using net with one output + multimodal soft attention
class SmallNetSoftAttentionFusion(nn.Module):
    def __init__(self,audio_dim,image_dim,out_dim):
        super(SmallNetSoftAttentionFusion, self).__init__()
        #video frames part
        self.frame_attention = nn.Linear(image_dim, 1)
        self.frame_transform = nn.Linear(image_dim, out_dim)


        #fusion part
        self.audio_transform = nn.Linear(audio_dim, out_dim)
        self.image_transform= nn.Linear(image_dim, out_dim)

        self.cross_modality_attention = nn.Linear(image_dim+audio_dim,2)

    def forward(self,audio_input,video_input):
        batch_reshaped = video_input.view(-1,512)
        attentions = self.frame_attention(batch_reshaped)
        batch_back = attentions.view(video_input.shape[0],video_input.shape[1],1) #reshape back to the batch,frames
        batch_back = batch_back.squeeze(-1)
        attentions = nn.functional.softmax(batch_back, dim=1)     
        attentions = attentions.unsqueeze(-1) #to (80,75,1)
        transformed_video = self.frame_transform(batch_reshaped)
        transformed_video = transformed_video.view(video_input.shape[0],video_input.shape[1],512) #reshape back to the batch,frames
        weighted_frames = transformed_video*attentions 
        video_input = torch.mean(weighted_frames,dim=1) #(batch,embedding)


        audio_input = normalize(audio_input)
        video_input = normalize(video_input)


        multi_mod_attention = self.cross_modality_attention(torch.cat((audio_input,video_input),dim=1))
        multi_mod_attention = nn.functional.softmax(multi_mod_attention, dim=1)
        
        audio_emb_transformed = self.audio_transform(audio_input)
        face_emb_transformed = self.image_transform(video_input)


        weighted_audio_emb = audio_emb_transformed * multi_mod_attention[:,0].unsqueeze(1)
        weighted_face_emb = face_emb_transformed * multi_mod_attention[:,1].unsqueeze(1)
        emb = torch.add(weighted_audio_emb,weighted_face_emb)
        return emb

# attention for video frames using big net accros all frames + multimodal soft attention
class BigNetSoftAttentionFusion(nn.Module):
    def __init__(self,audio_dim,image_dim,out_dim):
        super(BigNetSoftAttentionFusion, self).__init__()
        #video frames part
        self.frame_attention = nn.Linear(image_dim*75, 75)
        self.frame_transform = nn.Linear(image_dim, out_dim)


        #fusion part
        self.audio_transform = nn.Linear(audio_dim, out_dim)
        self.image_transform= nn.Linear(image_dim, out_dim)

        self.cross_modality_attention = nn.Linear(image_dim+audio_dim,2)

    def forward(self,audio_input,video_input):
        batch_reshaped = video_input.view(-1,512)
        
        #different part to small net
        attentions = self.frame_attention(video_input.view(video_input.shape[0],-1))
        attentions = nn.functional.softmax(attentions, dim=1)     
        attentions = attentions.unsqueeze(-1) #to (80,75,1)
        #different part to small net

        transformed_video = self.frame_transform(batch_reshaped)
        transformed_video = transformed_video.view(video_input.shape[0],video_input.shape[1],512) #reshape back to the batch,frames
        weighted_frames = transformed_video*attentions 
        video_input = torch.mean(weighted_frames,dim=1) #(batch,embedding)


        audio_input = normalize(audio_input)
        video_input = normalize(video_input)


        multi_mod_attention = self.cross_modality_attention(torch.cat((audio_input,video_input),dim=1))
        multi_mod_attention = nn.functional.softmax(multi_mod_attention, dim=1)
        
        audio_emb_transformed = self.audio_transform(audio_input)
        face_emb_transformed = self.image_transform(video_input)


        weighted_audio_emb = audio_emb_transformed * multi_mod_attention[:,0].unsqueeze(1)
        weighted_face_emb = face_emb_transformed * multi_mod_attention[:,1].unsqueeze(1)
        emb = torch.add(weighted_audio_emb,weighted_face_emb)
        return emb

# the main class of the audio visual model
class AudioVisual(nn.Module):
    def __init__(self,pretrained_wavlm=None,preatrained_vit=None,joint_training = False,frame_fusion=False):
        super(AudioVisual, self).__init__()

        if(frame_fusion == False): #if the model is frame-wise cross modality attention init different spk model
            self.speaker_extractor_model = videoSpkEncoder.WavLM_Base_MHFA(model_path=pretrained_wavlm,pooling="MHFA",head_nb=64,embed_dim=256,group=None,cnn_scale=0.0,layer_drop=0.00)
        else:
            self.speaker_extractor_model = videoSpkEncoder.WavLM_Base_MHFA_fusion(model_path=pretrained_wavlm,pooling="MHFA",head_nb=64,embed_dim=256,group=None,cnn_scale=0.0,layer_drop=0.00,joint_training=joint_training)


        checkpoint = torch.load(pretrained_wavlm)
        checkpoint.pop("projection.weight")
        self.speaker_extractor_model.load_state_dict(checkpoint,strict=False)
        self.joint_training = joint_training
        self.frame_fusion = frame_fusion

        if frame_fusion == False:
            if(joint_training == False):# if the joint training is on turn of random events in embedding extractors
                self.speaker_extractor_model.eval()
                self.vit_face_model = InceptionResnetV1(pretrained='vggface2').eval()
            else:
                self.vit_face_model = InceptionResnetV1(pretrained='vggface2')

            ## choose one of the fusion architectures
            # self.fusion = ConcatenationFusion(256,512,512)
            self.fusion = MeanSoftAttentionFusion(256,512,512)
            # self.fusion = SmallNetSoftAttentionFusion(256,512,512)
            # self.fusion = BigNetSoftAttentionFusion(256,512,512)
        else:
            self.vit_face_model = InceptionResnetV1(pretrained='vggface2').eval()



    def forward(self,audio_input,video_input):
        #input is batches  of audio adn videos
        #audio is loaded wav mono
        #images: [batch,frames,3,112,112]
        if self.frame_fusion == False:
            if(self.joint_training == False):
                with torch.no_grad(): #if the joint training is of dont track the gradients for embedding extractors
                    audio_emb = self.speaker_extractor_model(audio_input)

                    video_reshaped = video_input.view(-1,3,112,112) #concatenated frames of each sampel so i can /feed it into vit at once
                    face_emb = self.vit_face_model(video_reshaped)
                    face_emb = face_emb.view(video_input.shape[0],video_input.shape[1],512) #reshape back to the batch,frames,emb format

            else:
                audio_emb = self.speaker_extractor_model(audio_input)

                video_reshaped = video_input.view(-1,3,112,112) #concatenated frames of each sampel so i can /feed it into vit at once
                face_emb = self.vit_face_model(video_reshaped)
                face_emb = face_emb.view(video_input.shape[0],video_input.shape[1],512) #reshape back to the batch,frames,emb format

            emb = self.fusion(audio_emb,face_emb)
        #if the model is frame-wise cross modality attention redirect the fusion to speaker encoder
        else:
            if(self.joint_training == False):
                with torch.no_grad(): #if the joint training is of dont track the gradients for embedding extractors
                    video_reshaped = video_input.view(-1,3,112,112) #concatenated frames of each sampel so i can /feed it into vit at once
                    face_emb = self.vit_face_model(video_reshaped)
                    face_emb = face_emb.view(video_input.shape[0],video_input.shape[1],512) #reshape back to the batch,frames,emb format
                emb = self.speaker_extractor_model(audio_input,face_emb)
            else:
                video_reshaped = video_input.view(-1,3,112,112) #concatenated frames of each sampel so i can /feed it into vit at once
                face_emb = self.vit_face_model(video_reshaped)
                face_emb = face_emb.view(video_input.shape[0],video_input.shape[1],512) #reshape back to the batch,frames,emb format
                emb = self.speaker_extractor_model(audio_input,face_emb)

        return emb


#model + loss computation layer
class TrainingModel(nn.Module):
    def __init__(self,parameters,number_of_people=1):
        super(TrainingModel, self).__init__()
        self.audio_vis_model = AudioVisual(pretrained_wavlm=parameters["audio_string"],preatrained_vit=parameters["video_string"],joint_training=parameters["joint_training"],frame_fusion=parameters["frame_fusion"])

        #choose one of the losses
        # self.loss_fc = SoftMaxCrossEntropyLoss(512,number_of_people)
        self.loss_fc = ArcFace(512,number_of_people,scale=30,margin=0.2)


    def forward(self,audio_input,video_input,labels):
        audio_input,video_input,labels = check_dummy(audio_input,video_input,labels)
        predictions = self.audio_vis_model(audio_input,video_input)#get the output embeddings
        labels = torch.argmax(labels, dim=1)#conver one-hot labels to index labels

        loss = self.loss_fc(predictions,labels)#compute losses
        return loss
    
    #returns embedding without the losses, usefull for evaluation
    def get_embedding(self,audio_input,video_input):
        embedding = self.audio_vis_model(audio_input,video_input)
        return embedding

#class of the traiing procces
class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 gpu_id: int,
                 trial = None,
                 parameters = None,
                 data_augmentation = False
                 ):
        # self.gpu_id = gpu_id
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model,device_ids=[self.gpu_id],find_unused_parameters=True)
        self.batch_size = parameters["batch_size"]
        self.trial = trial

        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = parameters["save_every"]
        self.iteration_already = 0
        self.data_augmentation = data_augmentation

    def _run_batch(self, audio,images, targets):
        self.optimizer.zero_grad()
        loss = self.model(audio,images,targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):
        if self.gpu_id == 0:
            with open("training0/training_log.txt","a") as f:
                f.write(f"{time.ctime(time.time())}: starting epoch {epoch}\n")

        self.train_data.sampler.set_epoch(epoch)
        epoch_loss = torch.tensor(0)

        for (audio,image), targets in self.train_data:
            if self.gpu_id == 0:
                self.iteration_already += 1

            batch_loss = self._run_batch(audio,image, targets)
            if self.gpu_id == 0 and self.global_rank == 0:
                print(f"iteration: {self.iteration_already}, batch_loss: {batch_loss}")
            new_loss = epoch_loss + batch_loss
            epoch_loss=new_loss


        #track the loss of the whole epoch
        all_reduce(epoch_loss,op=ReduceOp.SUM)
        if(self.gpu_id == 0):
            with open("training0/training_log.txt","a") as f:
                f.write(f"{time.ctime(time.time())}  [GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(self.train_data)}, loss of whole epoch: {epoch_loss} \n")

            
        

    def _save_checkpoint(self, epoch, fraction = None):
        ckp = self.model.module.state_dict()
        PATH = f"training0/checkpoint{epoch}.pt"

        torch.save({'epoch':epoch,
                    'model_state' : ckp,
                    'optimizer_state' : self.optimizer.state_dict(),
                    }, PATH)
        with open("training0/training_log.txt","a") as f:
            f.write(f"Epoch {epoch} | Training checkpoint saved at {PATH}\n")


    def train(self, max_epochs: int):
        for epoch in range(max_epochs):

            self._run_epoch(epoch) #run epoch

            #save checkpoitn
            if self.gpu_id == 0 and self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

            #run verification
            if self.gpu_id == 0:
                with torch.no_grad():
                    # look how the model perform on verification dataset
                    # !!! this has no effect on the training !!! (just to help find the best model after the training is done)
                    eer = verificate(self.model,self.data_augmentation)
                    with open("training0/training_log.txt","a") as f:
                        f.write(f"Epoch {epoch} | EER on verification {eer} %\n")

                self.scheduler.step() #update learning rate
                with open("training0/training_log.txt","a") as f:
                    f.write(f"LR in next epoch: {self.scheduler.get_last_lr()[0]}\n")

#init all training objests
#training dataset, training model, optimizer,validation trial(None),scheduler
def load_train_objs(parameters):
    train_set = CustomDataset(parameters["audio_dataset_path"],parameters["images_dataset_path"],parameters["data_augmentation"],parameters["musan_path"])
    model = TrainingModel(number_of_people=train_set.persons,parameters=parameters)
    trial = None

    #if the models is training the frame-wise cross modal attention, the structure of the class model is different
    if(parameters["frame_fusion"]==False):
        param_groups = [ {'params': model.audio_vis_model.fusion.parameters()},
                        {'params': model.loss_fc.parameters()},] 
        #if joint training then add parameters of extractor models
        if(parameters["joint_training"] == True):
            #face weights
            face_layer_weights = [parameters["face_lr"] * (parameters["face_lr_factor"] ** i)  for i in range(parameters["face_layers"])]
            face_layer_names = [
                "conv2d_1a",
                "conv2d_2a",
                "conv2d_2b",
                "conv2d_3b",
                "conv2d_4a",
                "conv2d_4b",
                "repeat_1",
                "mixed_6a",
                "repeat_2",
                "mixed_7a",
                "repeat_3",
                "block8",
                "last_linear"]
            for layer_name,layer_lr in zip(face_layer_names,face_layer_weights):
                atrib = getattr(model.audio_vis_model.vit_face_model,layer_name)
                param_groups.append({'params': atrib.parameters(), 'lr': layer_lr})

            #audio weights
            param_groups.append({'params': model.audio_vis_model.speaker_extractor_model.back_end.parameters(), 'lr': parameters["mhfa_lr"]})
            for i in range(parameters["audio_layers"]):
                lr = parameters["audio_lr"] * (parameters["audio_lr_factor"] ** i)
                param_groups.append({'params': model.audio_vis_model.speaker_extractor_model.model.encoder.layers[i].parameters(), 'lr': lr})
    else:
        #the basic fusion models
        param_groups = [{'params': model.loss_fc.parameters()},
                        {'params': model.audio_vis_model.speaker_extractor_model.back_end.weights_quality},
                        {'params': model.audio_vis_model.speaker_extractor_model.back_end.attention_layer.parameters()},
                        {'params': model.audio_vis_model.speaker_extractor_model.back_end.image_transform.parameters()},
                        {'params': model.audio_vis_model.speaker_extractor_model.back_end.audio_transform.parameters()},
                        {'params': model.audio_vis_model.speaker_extractor_model.back_end.last_image_transform.parameters()},
                        ] 
        #if joint training then add parameters of extractor models
        if(parameters["joint_training"] == True):
            #face weights
            face_layer_weights = [parameters["face_lr"] * (parameters["face_lr_factor"] ** i)  for i in range(parameters["face_layers"])]
            face_layer_names = [
                "conv2d_1a",
                "conv2d_2a",
                "conv2d_2b",
                "conv2d_3b",
                "conv2d_4a",
                "conv2d_4b",
                "repeat_1",
                "mixed_6a",
                "repeat_2",
                "mixed_7a",
                "repeat_3",
                "block8",
                "last_linear"]
            for layer_name,layer_lr in zip(face_layer_names,face_layer_weights):
                atrib = getattr(model.audio_vis_model.vit_face_model,layer_name)
                param_groups.append({'params': atrib.parameters(), 'lr': layer_lr})
            
            #audio weights
            for i in range(parameters["audio_layers"]):
                lr = parameters["audio_lr"] * (parameters["audio_lr_factor"] ** i)
                param_groups.append({'params': model.audio_vis_model.speaker_extractor_model.model.encoder.layers[i].parameters(), 'lr': lr})
            param_groups.append({'params': model.audio_vis_model.speaker_extractor_model.back_end.weights_k, 'lr': parameters["mhfa_lr"]})
            param_groups.append({'params': model.audio_vis_model.speaker_extractor_model.back_end.weights_v, 'lr': parameters["mhfa_lr"]})
            param_groups.append({'params': model.audio_vis_model.speaker_extractor_model.back_end.cmp_linear_k.parameters(), 'lr': parameters["mhfa_lr"]})
            param_groups.append({'params': model.audio_vis_model.speaker_extractor_model.back_end.cmp_linear_v.parameters(), 'lr': parameters["mhfa_lr"]})
            param_groups.append({'params': model.audio_vis_model.speaker_extractor_model.back_end.att_head.parameters(), 'lr': parameters["mhfa_lr"]})
            param_groups.append({'params': model.audio_vis_model.speaker_extractor_model.back_end.pooling_fc.parameters(), 'lr': parameters["mhfa_lr"]})



    optimizer = optim.SGD(param_groups, lr=parameters["lr"],momentum=parameters["momentum"])
    scheduler = StepLR(optimizer, step_size=1, gamma=parameters["lr_decay"])
    return train_set,model, optimizer, trial,scheduler

def main(parameters):
    ddp_setup()
    
    dataset, model, optimizer, trial,scheduler = load_train_objs(parameters)
    data_loader = DataLoader(dataset,batch_size=parameters["batch_size"],shuffle=False,sampler=DistributedSampler(dataset,shuffle=True))
    trainer = Trainer(model=model,train_data= data_loader,optimizer=optimizer,scheduler=scheduler,gpu_id=None,trial=trial,parameters = parameters,data_augmentation=parameters["data_augmentation"])
    trainer.train(parameters["total_epochs"])
    destroy_process_group()

#main body
if __name__ == "__main__":
    #set all the possible seeds to assure determinism
    np.random.seed(42)
    np.random.RandomState(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    import random
    random.seed(123)

    parameters = {
        #path pretrained models
        "video_string" : "Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth",
        "audio_string" : "avg_model.pt",


        #datasets
        "audio_dataset_path" : "/flash/project_465000792/xbahou00/vox_1_dev",
        "images_dataset_path" : "/flash/project_465000792/xbahou00/vox_1_dev",
        "musan_path" : "/flash/project_465000792/xbahou00/musan",
         
        "data_augmentation" : False,

        "joint_training" : False,

        #hyper parameters
        "total_epochs" : 25,
        "save_every" : 1,
        "batch_size" : 80,

        #optimizer
        "lr": 0.005,
        "lr_decay": 0.7,
        "momentum" : 0.9,
        
        #visual model
        "face_lr" : 0.00001,
        "face_lr_factor" : 1.4,
        "face_layers" : 13,

        #audio model
        "audio_lr" : 0.00001,
        "audio_lr_factor" : 1.45,
        "audio_layers" : 12,
        "mhfa_lr" : 0.001,

        #utisls
        'logging_file': "training.log",


        #if True the programs train model that
        #compute cross modality weights for each frame 
        "frame_fusion" : False,
    }

    world_size = torch.cuda.device_count()

    with open("training0/training_log.txt","a") as f:
        f.write(f"{time.ctime(time.time())}: START\n")
    main(parameters)


