# Custom data loader for face images and audio 
# dataset paths have to be prealoaded using preloaded, then change the paths to the pickles

import os
from torch.utils.data import Dataset
import torch
import torchaudio
from torchvision import transforms
import time
import torchvision.transforms.functional as F
import pickle
import torchvision
from torchaudio.transforms import Resample
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



class CustomDataset(Dataset):
    def __init__(self,audio_dataset_path,images_dataset_path,data_augmentation=False,musan_path=None):
        self.desired_wav_length = 48_000  # 3 seconds of audio at 16 kHz sample rate
        self.data_person_id = {} #map person ID in dataset to my person ID 
        self.sample_id_to_path = {} #map sample ID to a path of sample
        self.sample_id_to_person = {} #map sample ID to a person ID ("for labels")
        self.audio_dataset_path = audio_dataset_path
        self.images_dataset_path = images_dataset_path
        self.persons = 0
        self.samples = 0
        self.transform = transforms.Compose([ 
                                                transforms.Normalize(0.5,0.5),
                                             ])
        self.resampler = Resample(orig_freq=44_100, new_freq=16_000)
        self.worker_id = torch.distributed.get_rank()

        self.data_augmentation = data_augmentation
        self.musan_path = musan_path
        self.musan_dict = {}
        if(data_augmentation == True):
            wav_paths = [] #load in paths to musan sounds
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
                self.musan_dict[i] = path
        self.blur = transforms.GaussianBlur(7,(0.1,3))
        self.crop = transforms.Compose([
            transforms.RandomCrop(size=(70,70)),
            transforms.Resize(size=(112,112))
        ])
        self.add_noice = torchaudio.transforms.AddNoise()
        self.rand_transform = RandomChoice([self.blur,self.crop])
        
        with open("training_log.txt","a") as f:
            f.write(f"{time.ctime(time.time())}: loading dataset\n")
        
        #rucne vlozim
        self.persons = 1211
        self.samples = 148642
        # self.persons = 3
        # self.samples = 389

        #load in pickles with paths to samples in dataset
        # Load dictionary from file
        with open('vox1_data_person_id.pkl', 'rb') as f:
            self.data_person_id = pickle.load(f)
        # Load dictionary from file
        with open('vox1_sample_id_to_path.pkl', 'rb') as f:
            self.sample_id_to_path = pickle.load(f)
            # Load dictionary from file
        with open('vox1_sample_id_to_person.pkl', 'rb') as f:
            self.sample_id_to_person = pickle.load(f)

        with open("training_log.txt","a") as f:
            f.write(f"{time.ctime(time.time())}: loaded dataset\n")


    def __len__(self):
        return self.samples
    
    def __getitem__(self, index):


        path = self.sample_id_to_path[index]


        try:
            wav_input, sample_rate = torchaudio.load(os.path.join(self.audio_dataset_path,path)) #load a tensor
            wav_input.to(self.worker_id)
            if(sample_rate == 44_100):
                wav_input = self.resampler(wav_input)
            length = wav_input.size(1)
            speech = wav_input[0][(length//2) - 24_000 :(length//2) + 24_000].to(self.worker_id)


        except Exception as e:
            print("chyba")
            return [torch.zeros(48_000).to(self.worker_id),torch.zeros(3,112,112).to(self.worker_id)], torch.zeros(self.persons).to(self.worker_id)


        if(path.split("/")[0] == "id11251"):
            print("missing id")
            return [torch.zeros(48_000).to(self.worker_id),torch.zeros(3,112,112).to(self.worker_id)], torch.zeros(self.persons).to(self.worker_id)

        if(path in ["id10618/MYfw_v7Cqcg/00035.wav","id10618/MYfw_v7Cqcg/00036.wav"]):
            print("missing sample")
            return [torch.zeros(48_000).to(self.worker_id),torch.zeros(3,112,112).to(self.worker_id)], torch.zeros(self.persons).to(self.worker_id)


        image = torchvision.io.read_image(os.path.join(self.images_dataset_path,f"{path[:-4]}.jpg")).to(self.worker_id)
        image = image / 255


        # DATA AUGMENTAION  #################################################
        if (self.data_augmentation == True):
            if(torch.rand(1) < 0.8): # augment in 80%
                #apply image augmentation
                if(torch.rand(1) < 0.5):
                    image = self.rand_transform(image)
                else:#apply audio augmentation
                    if(torch.rand(1) < 0.5): # add musan
                        wav_input, sample_rate = torchaudio.load(os.path.join(self.musan_path, self.musan_dict[int(torch.randint(low=0,high=len(self.musan_dict),size=(1,)))])) #load a tensor
                        wav_input.to(self.worker_id)
                        if(wav_input.size(1) >= 48_000):
                            noice =wav_input[0,(wav_input.size(1) // 2)- 24_000 : (wav_input.size(1) // 2)+24_000].to(self.worker_id)
                        else:
                            pad_amount = 48_000 - wav_input.size(1)
                            noice = torch.nn.functional.pad(wav_input[0], (0,pad_amount),value=0).to(self.worker_id)
                    else: # add random noice
                        noice = torch.rand_like(speech).to(self.worker_id)
                        noice = (noice * 2 )- 1
                    speech = self.add_noice(speech,noice,torch.tensor(1).to(self.worker_id))
            # #################################################



        image = self.transform(image)

        label = torch.zeros(self.persons).to(self.worker_id)
        label[self.sample_id_to_person[index]] = 1


        return [speech,image],label
