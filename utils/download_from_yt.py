#dowload videos from yt, and save them as utterances
#from each utterance save 3 second of wav at 16 kHz
# and 3 second of face images as a collage

import os
import pytube
import shutil
from moviepy.editor import VideoFileClip
import cv2
import time
import multiprocessing
import torchaudio
import torch
from torchaudio.transforms import Resample
import torchvision.transforms as transforms
import torchvision

#3 seconds of wav should have 132300 samples
#trim/pad if it doesnt match
def normalize_wav(wav,desired_length):
    current_length = len(wav)
    if current_length > desired_length:
        wav_input = wav[:desired_length]
    # If the current length is less than desired, pad the waveform
    elif current_length < desired_length:
        pad_amount = desired_length - current_length
        wav_input = torch.nn.functional.pad(wav, (0,pad_amount),value=0)
    return wav_input
def get_yt_id(filename):
    # Open the text file
    with open(filename, "r") as file:
        # Read all lines
        lines = file.readlines()

        # Initialize variable to store the Reference value
        reference_value = None

        # Iterate over each line
        for line in lines:
            # Split the line by the colon
            parts = line.split(":")

            # Check if the line contains the Reference value
            if parts[0].strip() == "Reference":
                # Extract the Reference value
                reference_value = parts[1].strip()
    return reference_value

def get_first_frame_and_bboxes(filename):
    first_frame = 0
    bboxes = {}
    # Open the text file
    with open(filename, "r") as file:
        # Read all lines
        lines = file.readlines()



        # Iterate over each line
        for i,line in enumerate(lines):
            if(i==7):
                parts = line.split()

                # Get the first element (number) from the split parts
                first_frame = int(parts[0])
            if(7 <= i and i < 7 + 25*3):
                parts = line.split()
                bboxes[int(parts[0])] = [int(parts[1]),int(parts[2]),int(parts[3]),int(parts[4])]
            elif(i >= 7 + 25*3 ):
                return first_frame,bboxes

def download_and_extract(id_folder):
    print(id_folder)
    HEIGHT = 360
    text_vid = "txt_test"  #path to folder with bounding boxes and video urls
    output_folder = "vox_1_test"
    frame_rate = 25
    duration = 3 #seconds
    age_restricted = []
    others_unavailable = []

    resampler = Resample(orig_freq=44_100, new_freq=16_000)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112,112)),
    ])


    os.makedirs(os.path.join(output_folder,id_folder))
    for video_folder in os.listdir(os.path.join(text_vid,id_folder)):
        os.makedirs(os.path.join(output_folder,id_folder,video_folder))
        yt_id = ""
        #get the link
        for video_text in os.listdir(os.path.join(text_vid,id_folder,video_folder)):
            yt_id = get_yt_id(os.path.join(text_vid,id_folder,video_folder,video_text))
        video_url = "https://www.youtube.com/watch?v=" + yt_id


        yt = pytube.YouTube(video_url)

        try:
            # Select the highest resolution stream
            stream = yt.streams.get_highest_resolution()
        except Exception as e:
            others_unavailable.append(os.path.join(output_folder,id_folder,video_folder))
            continue

        # Download the video
        try:
            stream.download(os.path.join(output_folder,id_folder,video_folder),filename="video.mp4")
        except Exception as e:
            others_unavailable.append(os.path.join(output_folder,id_folder,video_folder))
            continue


        first_frames = []
        bboxes = []
        #get begginging frame
        for i in range(1,200):
            if os.path.exists(os.path.join(text_vid,id_folder,video_folder,"0" * (5-len(str(i))) + str(i) + ".txt")) == False:
                break
        # for video_text in os.listdir(os.path.join(text_vid,id_folder,video_folder)):
            first_frame,bbox = get_first_frame_and_bboxes(os.path.join(text_vid,id_folder,video_folder,"0" * (5-len(str(i))) + str(i) + ".txt"))
            first_frames.append(first_frame)
            bboxes.append(bbox)
        
        #loop thorough 
        video_clip = VideoFileClip(os.path.join(output_folder,id_folder,video_folder,"video.mp4"))
        video_audio = video_clip.audio
        video_audio.write_audiofile(os.path.join(output_folder,id_folder,video_folder,"tmp_audio.wav"))
        wav_input, sample_rate = torchaudio.load(os.path.join(output_folder,id_folder,video_folder,"tmp_audio.wav"))
        wav_input =torch.tensor((wav_input[0] + wav_input[1])/2)
        if(sample_rate == 44_100):
            wav_input = resampler(wav_input)

        (_,video_height) = video_clip.size

        n_videos = 0 #video segments done
        frames_of_segment = 0 #

        audios = []
        video_frames = []

        for frame_n,frame in enumerate(video_clip.iter_frames(fps=frame_rate)):
                
            folder_name = str(n_videos+1)
            zeros = "0" * (5-len(folder_name))
            folder_name = zeros + folder_name

            #create folder and load in a wav
            if (frame_n == first_frames[n_videos]):
                start_time = frame_n * 640
                end_time = start_time + 48_000 #3 vteriny
                video_audio_subclip = wav_input[start_time:end_time]
                audios.append(video_audio_subclip)

            if (first_frames[n_videos] <= frame_n  and frame_n <first_frames[n_videos]+duration*frame_rate):
                scale = video_height/HEIGHT
                x,y,w,h = bboxes[n_videos][frame_n]
                x,y,w,h = int(x*scale),int(y*scale),int(w*scale),int(h*scale)
                frame = frame[y:y+h,x:x+w]

                torch_tensor = transform(frame)
                video_frames.append(torch_tensor)

                frames_of_segment+=1

                if(frame_n == (first_frames[n_videos]+duration*frame_rate)-1):
                    n_videos += 1
                    frames_of_segment = 0 
                    if(n_videos >= len(first_frames)):
                        break
                       


        video_clip.close()
        os.remove(os.path.join(output_folder,id_folder,video_folder,"video.mp4"))
        os.remove(os.path.join(output_folder,id_folder,video_folder,"tmp_audio.wav"))

        if(video_frames == []):
            continue

        video_frames = torch.stack(video_frames)
        flattened_images = video_frames.view(-1, 3, 112, 112)
        for i in range(flattened_images.size(0)//75):
            zeros = 5 - len(str(i+1))
            image = flattened_images[i*75:(i+1)*75,:,:,:]
            grid = torchvision.utils.make_grid(image,nrow=75,padding=0)
            grid = grid.unsqueeze(0)
            torchvision.utils.save_image(grid,fp=os.path.join(output_folder,id_folder,video_folder,f"{zeros*'0'}{i+1}.jpg"))

        for i,audio in enumerate(audios):
            zeros = 5 - len(str(i))
            torchaudio.save(os.path.join(output_folder,id_folder,video_folder,f"{zeros*'0'}{i+1}.wav"), audio.unsqueeze(0), sample_rate=16_000) 
            


    return [age_restricted,others_unavailable]

if __name__ == "__main__":


    HEIGHT = 360

    text_vid = "txt_test"   #path to folder with bounding boxes and video urls
    output_folder = "vox_1_test"
    age_restricted_file = "age_rest_dev.txt"                
    others_unavailable_file = "others_unavailable_dev.txt"

    frame_rate = 25
    duration = 3 #seconds
    age_restricted = []
    others_unavailable = []
    lower_video_bound = 10_270 # inclusice
    upper_video_bound = 10_310 # exclusive


    folders = []
    for id_folder in os.listdir(text_vid):
        if(lower_video_bound <= int(id_folder[2:]) and int(id_folder[2:]) < upper_video_bound):
            folders.append(id_folder)
    done_folders = os.listdir(output_folder)
    done_folders = set(done_folders)
    folders = set(folders)
    folders = folders - done_folders
    folders = list(folders)

    num_processes = 3

    os.makedirs(output_folder,exist_ok=True)


    with multiprocessing.Pool(processes=num_processes) as pool:
        result = pool.map(download_and_extract, folders)

    for age,other in result:
        others_unavailable.extend(age_restricted)
        others_unavailable.extend(other)


    # Open the file in append mode
    with open(others_unavailable_file, 'a') as file:
        # Write content to the file
        for i in others_unavailable:
            file.write(i + "\n")
