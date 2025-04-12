# test model on voxceleb1 verification trial

import numpy as np
np.random.seed(42)
np.random.RandomState(42)
import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
import random
random.seed(123)
from simplecombined_model import AudioVisual,TrainingModel


import torchaudio
import os
import tarfile
import shutil
from sklearn.metrics import det_curve
import numpy as np
import torchvision
import pickle
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.functional import normalize 
from torchvision import transforms
from torchvision.transforms.v2 import RandomChoice

#trim/pad if the length of audio doesnt match
def normalize_wav(wav,desired_length):
    current_length = len(wav)
    if current_length > desired_length:
        wav_input = wav[:desired_length]
    # If the current length is less than desired, pad the waveform
    elif current_length < desired_length:
        pad_amount = desired_length - current_length
        wav_input = torch.nn.functional.pad(wav, (0,pad_amount),value=0)
    return wav_input


#load in batch of input data and augment it
def noice_image_and_wav_loader(paths,audio_dataset_path,video_dataset_path,musan_path,musan_dict):
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
            image = torchvision.io.read_image(os.path.join(video_dataset_path,f"{path[:-4]}.jpg"))
        except:
            print("img load fail")
            return None
        image = image / 255
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        try:
            speech, sample_rate = torchaudio.load(os.path.join(audio_dataset_path,path))
        except:
            print(os.path.join(audio_dataset_path,path))
            print("audio load fail")
            return None
        speech = speech[0][ (len(speech[0]) // 2) - 24_000: (len(speech[0]) // 2) + 24_000]


        # DATA AUGMENTAION  2 #################################################
        if(torch.rand(1) < 0.8): # augment in 80%
            #apply image augmentation
            if(torch.rand(1) < 0.5):
                image = rand_transform(image)
            else:#apply audio augmentation
                if(torch.rand(1) < 0.5): # add musan
                    wav_input, sample_rate = torchaudio.load(os.path.join(musan_path, musan_dict[int(torch.randint(low=0,high=len(musan_dict),size=(1,)))])) #load a tensor
                    if(wav_input.size(1) >= 48_000):
                        noice =wav_input[0,(wav_input.size(1) // 2)- 24_000 : (wav_input.size(1) // 2)+24_000]
                    else:
                        pad_amount = 48_000 - wav_input.size(1)
                        noice = torch.nn.functional.pad(wav_input[0], (0,pad_amount),value=0)
                else: # add random noice
                    noice = torch.rand_like(speech)
                    noice = (noice * 2 )- 1
                speech = add_noice(speech,noice,torch.tensor(1))
        # #################################################



        img=transform(image)
        audios.append(speech)
        images.append(img)
    images = torch.stack(images).to(0)
    audios = torch.stack(audios).to(0)
    return images,audios

#load in batch of input data
def image_and_wav_loader(paths,audio_dataset_path,video_dataset_path):
    # vox_1_test_untarred\id10298\f5eaTNf7-io\00002.wav
    images = []
    audios = []
    for path in paths:
        try:
            image = torchvision.io.read_image(os.path.join(video_dataset_path,f"{path[:-4]}.jpg"))
        except:
            print("img load fail")
            return None
        image = image / 255
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        try:
            speech, sample_rate = torchaudio.load(os.path.join(audio_dataset_path,path))
        except:
            print(os.path.join(audio_dataset_path,path))
            print("audio load fail")
            return None
        speech = speech[0][ (len(speech[0]) // 2) - 24_000: (len(speech[0]) // 2) + 24_000]


        img=transform(image)
        audios.append(speech)
        images.append(img)
    images = torch.stack(images).to(0)
    audios = torch.stack(audios).to(0)
    return images,audios

#get embedding for evaluation
def get_audio_vis_embs(images,audios,model):
    #    path                       dataset_path 
#    id10270/8jEAjG6SegY/00018.wav vox_1_test_untarred

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

#compute EER on DET
def compute_eer(fpr,fnr):
    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
    eer = fnr[x1] + a * (fnr[x2] - fnr[x1])
    return eer

def compute_min_dcf(fpr, fnr, p_target, c_miss=1, c_fa=1):
    min_det = min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))
    return min_det


# load in paths to musan samples to dictionary
musan_path = "/flash/project_465000792/xbahou00/musan"
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

id_2_musan = {}
for i,path in enumerate(wav_paths):
    id_2_musan[i] = path




audio_dataset_path = "/flash/project_465000792/xbahou00/vox1_test/test/wav"
video_dataset_path = "/flash/project_465000792/xbahou00/vox1test_faces"
musan_path = "/flash/project_465000792/xbahou00/musan"

audio_model_path = "avg_model.pt"
model_path = "training35/checkpoint0.pt"

trials_path = "voxceleb1pairs.ssv"

# uncomment if you want the concatenation
####################################################################
# #            CONC
# face_model = InceptionResnetV1(pretrained='vggface2').eval().to(0)
# audio_model = WavLM_Base_MHFA(audio_model_path,"MHFA",64,256,None,0,0)
# checkpoint = torch.load(audio_model_path)
# checkpoint.pop("projection.weight")
# audio_model.load_state_dict(checkpoint)
# audio_model.eval().to(0)
####################################################################

#comment out if you want concatenation
# # my combined #######################################################
checkpoint = torch.load(model_path)
training_model  = TrainingModel({"audio_string":"avg_model.pt","video_string":None,"joint_training":False},number_of_people=1211)
training_model.load_state_dict(checkpoint["model_state"],strict=False)
model = training_model.audio_vis_model.to(0)
# # my combined #######################################################



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

batch_size = 10
path2emb = {}
for i in range(0,len(all_paths),batch_size):
    j = i + batch_size if ((i + batch_size) <len(all_paths) ) else len(all_paths)

    # choose if augment or not
    # images,audios = noice_image_and_wav_loader(all_paths[i:j],audio_dataset_path,video_dataset_path,musan_path,id_2_musan)
    images,audios = image_and_wav_loader(all_paths[i:j],audio_dataset_path,video_dataset_path)

    #comment out if concatenation fusion
    ######################################################
    #           FUSION
    embs = get_audio_vis_embs(images,audios,model)

    if(embs == None):
        print("None")
        continue
    ######################################################

    # uncomment if you want concatenation fusion
    #################################################
    # # ##     CONC
    # with torch.no_grad():
    #     emb1 = audio_model(audios)
    #     emb2 = face_model(images)

    # if(emb1 == None or emb2 == None):
    #     continue
    
    # emb1 = normalize(emb1)
    # emb2 = normalize(emb2)
    # # embs = emb2
    # embs = torch.cat((emb1,emb2),dim=1)
    #################################################

    for i,path in enumerate(all_paths[i:j]):
        path2emb[path] = embs[i]
    print("one")
# Save dictionary to file
with open('vox_1_test_one_image.pkl', 'wb') as f:
    pickle.dump(path2emb, f)

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



def comp_cos_sim(embs1,embs2):
    top = (embs1 * embs2).sum(dim=1)
    A = torch.norm(embs1,dim=1)
    B = torch.norm(embs2,dim=1)
    bottom = torch.mul(A,B)
    return torch.div(top,bottom)



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



cos_similarities = cos_similarities.cpu()
fpr, tpr, thresholds = roc_curve(real_labels, cos_similarities)

eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)

print(f"EER: {eer*100} %")
fpr, fnr, thresholds = det_curve(real_labels, cos_similarities)
min_dcf,min_dcf_thresh = ComputeMinDcf(fnr,fpr,thresholds,0.01,1,1)
print(f"min dcf: {min_dcf}, min_dcf threshold: {min_dcf_thresh}")
print(model_path)


