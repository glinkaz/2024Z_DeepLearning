import os
import csv
from datasets import Dataset
import librosa
import torch
import numpy as np
from torch import tensor
from jiwer import cer, wer
import Levenshtein
from transformers import TrainerCallback, TrainerControl

labels_to_classify = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']


def predict_single_audio(model, processor, audio_file_path):
    audio, rate = librosa.load(audio_file_path, sr=16000)
    inputs = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).to('cuda')
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    # logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    print(f"Transcription for {audio_file_path}: {transcription}")
    return transcription


def predict_on_dataset(model, processor, dataset):
    predictions = []
    ground_truths = []

    for data in dataset:
        inputs = tensor([dataset[0]['input_values']]).to('cuda')
        with torch.no_grad():
            logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        ground_truths.append(processor.decode(data['labels'][0]))
        predictions.append(transcription)

    cer_score = cer(ground_truths, predictions)
    accuracy = sum(1 for true, pred in zip(ground_truths, predictions) if true == pred) / len(ground_truths)

    print("CER:", cer_score)
    print("Accuracy:", accuracy)
    return cer, accuracy


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
        if len(inputs.input_values[0]) != 16000:
            print(audio_dir, audio_file, len(inputs.input_values[0]))
        input_values.append(inputs.input_values[0])

        label = os.path.basename(os.path.dirname(audio_file)).upper()
        label = pad_string(label, 6, '<pad>')
        with processor.as_target_processor():
            label = processor(label, return_tensors="pt", padding=True).input_ids
        labels.append(label)

    column_data = {"input_values": input_values, "labels": labels}
    dataset = Dataset.from_dict(column_data)
    return dataset


def get_available_labels(directory):
    available_labels = [name.upper() for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return available_labels


def compute_metrics_with_closest_label(available_labels, predictions, labels):
    correct_count = 0
    total_count = len(predictions)
    total_cer = 0

    for prediction, label in zip(predictions, labels):
        closest_label = min(available_labels, key=lambda l: Levenshtein.distance(l, prediction))
        if closest_label == label:
            correct_count += 1
        total_cer += cer(label, closest_label)

    accuracy = correct_count / total_count
    cer_score = total_cer / total_count
    return cer_score, accuracy


class SaveMetricsCallback(TrainerCallback):
    def __init__(self, validation_dataset, name=''):
        self.validation_dataset = validation_dataset
        self.metrics_filename = f"validation_metrics_{name}.csv"
        self.fieldnames = ["epoch", "accuracy", "cer", "accuracy_closest", "cer_closest"]

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        accuracy = metrics["eval_accuracy"]
        cer = metrics["eval_cer"]
        accuracy_closest = metrics["eval_accuracy_closest"]
        cer_closest = metrics["eval_cer_closest"]
        epoch = state.epoch

        with open(self.metrics_filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if file.tell() == 0:  
                writer.writeheader()
            writer.writerow({"epoch": epoch, "accuracy": accuracy, "cer": cer, "accuracy_closest": accuracy_closest, "cer_closest": cer_closest})
