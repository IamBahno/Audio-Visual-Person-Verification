#preload dataset, meaning save paths and ids to a pickle

import os
import tarfile
import pickle
import torchaudio

dataset_path = "/scratch/project_465000792/oplchot/data/raw_data2/voxceleb1/dev/wav"
data_person_id = {} #map person ID in dataset to my person ID 
sample_id_to_path = {} #map sample ID to a path of sample
sample_id_to_person = {} #map sample ID to a person ID ("for labels")
dataset_path = dataset_path
persons = 0
samples = 0
for id_folder in os.listdir(dataset_path):

    dataset_user_id = int(id_folder[2:])
    data_person_id[dataset_user_id] = persons

    for video in os.listdir(os.path.join(dataset_path,id_folder)):
        for utterance in os.listdir(os.path.join(dataset_path,id_folder,video)):

            try:
                wav_input, sample_rate = torchaudio.load(os.path.join(dataset_path,id_folder,video,utterance))
                if(wav_input.size(1) < 48_000):
                    print("short")
                    continue
            except:
                print(dataset_path,id_folder,video)
                continue
            sample_id_to_path[samples] = os.path.join(id_folder,video,utterance)
            sample_id_to_person[samples] = persons
            samples+=1                     

    persons += 1
    print(persons)


# Save dictionary to file
with open('vox1_data_person_id.pkl', 'wb') as f:
    pickle.dump(data_person_id, f)

# Save dictionary to file
with open('vox1_sample_id_to_path.pkl', 'wb') as f:
    pickle.dump(sample_id_to_path, f)

# Save dictionary to file
with open('vox1_sample_id_to_person.pkl', 'wb') as f:
    pickle.dump(sample_id_to_person, f)

with open('vox1_meta.txt', 'w') as f:
    f.write(f"persons: {persons}\n")
    f.write(f"samples: {samples}\n")