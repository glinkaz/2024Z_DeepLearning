import os

import librosa
import torch
from datasets import Dataset


def predict_single_audio(model, processor, audio_file_path):
    audio, rate = librosa.load(audio_file_path, sr=16000)
    inputs = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000)

    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    print(f"Transcription for {audio_file_path}: {transcription}")
    return transcription


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
        audio_path = os.path.join(audio_dir, audio_file)

        audio, rate = librosa.load(audio_path, sr=16000)
        inputs = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000)
        label = os.path.basename(os.path.dirname(audio_file)).upper()
        label = pad_string(label, 6, '<pad>')
        print('s')
        with processor.as_target_processor():
            label = processor(label, return_tensors="pt", padding=True).input_ids
        input_values.append(inputs.input_values[0])
        labels.append(label)

    column_data = {"input_values": input_values, "labels": labels}
    dataset = Dataset.from_dict(column_data)
    return dataset
