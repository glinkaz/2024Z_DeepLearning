import os
from random import random, shuffle

import librosa
import numpy as np
from datasets import Dataset


def create_silence_detection_list(audio_dir='./data/train/audio/', output_file='./data/train/silence_detection_list.txt'):
    folders = [folder for folder in os.listdir(audio_dir) if folder not in ['_background_noise_', '.DS_Store']]
    with open(output_file, 'w') as f:
        for folder in folders:
            files = os.listdir(os.path.join(audio_dir, folder))
            if folder == 'silence':
                selected_files = files
            else:
                selected_files = random.sample(files, min(80, len(files)))
            for file in selected_files:
                f.write(f'{folder}\\{file}\n')


def train_test_split_silence_detection(file_list='./data/train/silence_detection_list.txt', test_ratio=0.2):
    with open(file_list, 'r') as f:
        lines = f.readlines()
    shuffle(lines)
    split_idx = int(test_ratio * len(lines))
    test_lines = lines[:split_idx]
    train_lines = lines[split_idx:]
    with open('./data/train/train_silence_detection_list.txt', 'w') as f:
        f.writelines(train_lines)
    with open('./data/train/test_silence_detection_list.txt', 'w') as f:
        f.writelines(test_lines)


def pad_string(s, target_length, fillstr):
    fill_count = target_length - len(s)
    s += fillstr * fill_count
    return s


def create_dataset(audio_dir, processor, file_list):
    input_values = []
    labels = []
    with open(f'data/train/{file_list}.txt', 'r') as file:
        training_list = [line.rstrip() for line in file]

    for audio_file in training_list:
        audio, rate = librosa.load(os.path.join(audio_dir, audio_file), sr=16000)
        if len(audio) < 16000:
            padding = np.zeros(16000 - len(audio))
            audio = np.concatenate((audio, padding))
        inputs = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000)
        input_values.append(inputs.input_values[0])

        label = os.path.basename(os.path.dirname(audio_file)).upper()
        label = pad_string(label, 6, '<pad>')
        with processor.as_target_processor():
            label = processor(label, return_tensors="pt", padding=True).input_ids
        labels.append(label)

    column_data = {"input_values": input_values, "labels": labels}
    dataset = Dataset.from_dict(column_data)
    return dataset


def create_dataset_silence_classification(audio_dir, processor, file_list):
    input_values = []
    labels = []
    with open(f'data/train/{file_list}.txt', 'r') as file:
        training_list = [line.rstrip() for line in file]

    for audio_file in training_list:
        audio, rate = librosa.load(os.path.join(audio_dir, audio_file), sr=16000)
        if len(audio) < 16000:
            padding = np.zeros(16000 - len(audio))
            audio = np.concatenate((audio, padding))
        inputs = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000)
        input_values.append(inputs.input_values[0])

        folder_name = os.path.basename(os.path.dirname(audio_file))
        if folder_name == 'silence':
            label = '<pad>' * 5
        else:
            label = 'SOUND'
        with processor.as_target_processor():
            label = processor(label, return_tensors="pt", padding=True).input_ids
        labels.append(label)

    column_data = {"input_values": input_values, "labels": labels}
    dataset = Dataset.from_dict(column_data)
    return dataset