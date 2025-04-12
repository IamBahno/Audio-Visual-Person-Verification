#the main training script for models fusion audio and face embedding
# in the main function: set paths to models and datasets
#                      choose if data should be augmented
#                      choose if to joint train with the embeddings extractors
# in class AudioVisual(nn.Module) choose which fusion model to use
# before running: create folder "training0" fot logging and saving models
#                 in DatasetLoader: set number of identities and samples in dataset
#                                    and paths to pickles with paths of samples

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from simpleSpkEncoder import WavLM_Base_MHFA
import torch.optim as optim
from torch.utils.data import DataLoader
from simpleDatasetLoader import CustomDataset
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
        try:
            image = torchvision.io.read_image(os.path.join(video_dataset_path,f"{path[:-4]}.jpg")).to(0)
        except:
            print("img load fail")
            return None
        image = image / 255
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        try:
            speech, sample_rate = torchaudio.load(os.path.join(audio_dataset_path,path))
            speech = speech.to(0)
        except:
            print(os.path.join(audio_dataset_path,path))
            print("audio load fail")
            return None
        speech = speech[0][ (len(speech[0]) // 2) - 24_000: (len(speech[0]) // 2) + 24_000].to(0)


        # DATA AUGMENTAION  2 #################################################
        if(torch.rand(1) < 0.8): # augment in 80%
            #apply image augmentation
            if(torch.rand(1) < 0.5):
                image = rand_transform(image)
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
        try:
            img = torchvision.io.read_image(os.path.join(video_dataset_path,f"{path[:-4]}.jpg"))
        except:
            print("img load fail")
            return None
        img = img / 255
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        img=transform(img)
        images.append(img)
        try:
            wav_input, sample_rate = torchaudio.load(os.path.join(audio_dataset_path,path))
        except:
            print(os.path.join(audio_dataset_path,path))
            print("audio load fail")
            return None
        wav_input = wav_input[0][ (len(wav_input[0]) // 2) - 24_000: (len(wav_input[0]) // 2) + 24_000]
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
    audio_dataset_path = "/flash/project_465000792/xbahou00/vox1_test/test/wav"
    video_dataset_path = "/flash/project_465000792/xbahou00/vox1test_faces"
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
            images,audios = image_and_wav_loader(all_paths[i:j],audio_dataset_path,video_dataset_path)
        else:
            images,audios = data_augmented_image_and_wav_loader(all_paths[i:j],audio_dataset_path,video_dataset_path,musan_path,musan_dict)


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

#soft attention fusion
class Fusion(nn.Module):
    def __init__(self,audio_dim,image_dim,out_dim):
        super(Fusion, self).__init__()
        #fusion part
        self.audio_transform = nn.Linear(audio_dim, out_dim)
        self.image_transform= nn.Linear(image_dim, out_dim)

        self.cross_modality_attention = nn.Linear(image_dim+audio_dim,2)

    def forward(self,audio_input,video_input):
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

#fusion with 3 neural nets (its not necessarily linear)
class LinearLayersFusion(nn.Module):
    def __init__(self,audio_dim,image_dim,out_dim):
        super(LinearLayersFusion, self).__init__()
        self.linear1 = nn.Linear(audio_dim+image_dim, audio_dim+image_dim)
        self.linear2 = nn.Linear(audio_dim+image_dim, out_dim)
        self.linear3 = nn.Linear(out_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(audio_dim+image_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self,audio_input,video_input):
        audio_input = normalize(audio_input)
        video_input = normalize(video_input)
        
        x =  self.relu(self.bn1(self.linear1(torch.cat((audio_input,video_input),dim=1))))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        return x

#gated multi modal fusion
class GatedFusion(nn.Module):
    def __init__(self,audio_dim,image_dim,out_dim):
        super(GatedFusion, self).__init__()
        self.audio_transform = nn.Linear(audio_dim, out_dim)
        self.image_transform= nn.Linear(image_dim, out_dim)

        self.joint_fusion_1= nn.Linear(image_dim+audio_dim, 32)
        self.joint_fusion_2= nn.Linear(32, out_dim)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,audio_input,video_input):
        audio_input = normalize(audio_input)
        video_input = normalize(video_input)

        joint_embdeddings = self.joint_fusion_1(torch.cat((audio_input,video_input),dim=1))
        joint_embdeddings = self.relu(self.bn(joint_embdeddings))
        joint_embdeddings = self.joint_fusion_2(joint_embdeddings)
        joint_embdeddings = self.sigmoid(joint_embdeddings)


        audio_emb_transformed = torch.tanh(self.audio_transform(audio_input))
        face_emb_transformed = torch.tanh(self.image_transform(video_input))

        gated_audio = audio_emb_transformed * joint_embdeddings
        gated_video = face_emb_transformed * (1-joint_embdeddings)

        out = gated_audio.unsqueeze(0) + gated_video.unsqueeze(0)
        return out.squeeze()


#Fusion with lower level soft attentions
class LowerLevelSoftAttentionFusion(nn.Module):
    def __init__(self,audio_dim,image_dim,out_dim,nlower_levels=3,audio_lower_dim=768,video_low_dim_1 = 2809,video_low_dim_2=576,video_low_dim_3=121):
        super(LowerLevelSoftAttentionFusion, self).__init__()
        self.inter_level_attention_1 = nn.Linear(audio_lower_dim+video_low_dim_1,2)
        self.inter_level_attention_2 = nn.Linear(audio_lower_dim+video_low_dim_2,2)
        self.inter_level_attention_3 = nn.Linear(audio_lower_dim+video_low_dim_3,2)
        self.embedding_attention = nn.Linear(audio_dim+image_dim,2)

        self.audio_transform = nn.Linear(audio_dim, out_dim)
        self.image_transform= nn.Linear(image_dim, out_dim)


    def forward(self,audio_input,video_input,audio_inter,video_inter):
        audio_input = normalize(audio_input)
        video_input = normalize(video_input)
        audio_emb_transformed = self.audio_transform(audio_input)
        face_emb_transformed = self.image_transform(video_input)

        inter_attention1 =self.inter_level_attention_1(torch.cat((audio_inter[0],video_inter[0]),dim=1))
        inter_attention2 =self.inter_level_attention_2(torch.cat((audio_inter[1],video_inter[1]),dim=1))
        inter_attention3 =self.inter_level_attention_3(torch.cat((audio_inter[2],video_inter[2]),dim=1))
        embed_attention = self.embedding_attention(torch.cat((audio_input,video_input),dim=1))

        attentions = torch.mean(torch.stack([inter_attention1,inter_attention2,inter_attention3,embed_attention]),dim=0)
        attentions = nn.functional.softmax(attentions, dim=1)

        weighted_audio_emb = audio_emb_transformed * attentions[:,0].unsqueeze(1)
        weighted_face_emb = face_emb_transformed * attentions[:,1].unsqueeze(1)
        emb = torch.add(weighted_audio_emb,weighted_face_emb)
        return emb


# the main class of the audio visual model
class AudioVisual(nn.Module):
    def __init__(self,pretrained_wavlm=None,preatrained_vit=None,joint_training = False):
        super(AudioVisual, self).__init__()

        self.speaker_extractor_model = WavLM_Base_MHFA(model_path=pretrained_wavlm,pooling="MHFA",head_nb=64,embed_dim=256,group=None,cnn_scale=0.0,layer_drop=0.00)

        checkpoint = torch.load(pretrained_wavlm)
        checkpoint.pop("projection.weight")
        self.speaker_extractor_model.load_state_dict(checkpoint)
        if(joint_training == False):
            self.speaker_extractor_model.eval()
            self.vit_face_model = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            self.vit_face_model = InceptionResnetV1(pretrained='vggface2')
        self.joint_training = joint_training

        #choose one of the fusion architectures
        # self.fusion = Fusion(256,512,512)
        # self.fusion = LinearLayersFusion(256,512,512)
        self.fusion = GatedFusion(256,512,512)
        # self.fusion = LowerLevelSoftAttentionFusion(256,512,512)


    def forward(self,audio_input,video_input):
        #input is batches  of audio and iamges
        #audio is loaded wav mono
        #images: [batch,3,112,112]

        #if the fusion is LowerLevelSoftAttentionFusion it need retrieve those level level representattions
        if(isinstance(self.fusion,LowerLevelSoftAttentionFusion)):
            if(self.joint_training == False):   
                with torch.no_grad(): #if the joint training is of dont track the gradients for embedding extractors
                    audio_emb,audio_inters = self.speaker_extractor_model.get_emb_and_lower_levels(audio_input)
                    face_emb,face_inters= self.vit_face_model.get_emb_and_lower_intermediets(video_input)
            else:
                audio_emb,audio_inters = self.speaker_extractor_model.get_emb_and_lower_levels(audio_input)
                face_emb,face_inters= self.vit_face_model.get_emb_and_lower_intermediets(video_input)
            emb = self.fusion(audio_emb,face_emb,audio_inters,face_inters)
        else:# else use the standart forward functions
            if(self.joint_training == False):
                with torch.no_grad(): #if the joint training is of dont track the gradients for embedding extractors
                    audio_emb = self.speaker_extractor_model(audio_input)
                    face_emb = self.vit_face_model(video_input)

            else:
                audio_emb = self.speaker_extractor_model(audio_input)
                face_emb = self.vit_face_model(video_input)
            emb = self.fusion(audio_emb,face_emb)
        

        return emb


#model + loss computation layer
class TrainingModel(nn.Module):
    def __init__(self,parameters,number_of_people=1):
        super(TrainingModel, self).__init__()
        self.audio_vis_model = AudioVisual(pretrained_wavlm=parameters["audio_string"],preatrained_vit=parameters["video_string"],joint_training=parameters["joint_training"])

        #choose one of the losses
        # self.loss_fc = SoftMaxCrossEntropyLoss(512,number_of_people)
        self.loss_fc = ArcFace(512,number_of_people,scale=30,margin=0.2)


    def forward(self,audio_input,video_input,labels):
        audio_input,video_input,labels = check_dummy(audio_input,video_input,labels)
        predictions = self.audio_vis_model(audio_input,video_input) #get the output embeddings
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

        #loop thorugh dataset
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
                        f.write(f"Epoch {epoch} | EER on verification {eer}%\n")

                self.scheduler.step() #update learning rate
                with open("training0/training_log.txt","a") as f:
                    f.write(f"LR in next epoch: {self.scheduler.get_last_lr()[0]}\n")

#init all training objests
#training dataset, training model, optimizer,validation trial(None),scheduler
def load_train_objs(parameters):
    train_set = CustomDataset(parameters["audio_dataset_path"],parameters["images_dataset_path"],parameters["data_augmentation"],parameters["musan_path"])
    model = TrainingModel(number_of_people=train_set.persons,parameters=parameters)
    trial = None #just legacy line noew
    param_groups = [ {'params': model.audio_vis_model.fusion.parameters()},
                    {'params': model.loss_fc.parameters()},] 

    #if joint training then add parameters of extractor models and mhfa parameters
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
        "audio_dataset_path" : "/flash/project_465000792/xbahou00/wav",
        "images_dataset_path" : "/flash/project_465000792/xbahou00/vox1dev_faces",
        "musan_path" : "/flash/project_465000792/xbahou00/musan",
         
        "data_augmentation" : False,

        "joint_training" : False,

        #hyper parameters
        "total_epochs" : 20,
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
        'logging_file': "training.log"
    }

    world_size = torch.cuda.device_count()

    with open("training0/training_log.txt","a") as f:
        f.write(f"{time.ctime(time.time())}: START\n")
    main(parameters)


