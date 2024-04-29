import os
import csv
import librosa
import torch
from jiwer import cer
import Levenshtein
from transformers import TrainerCallback


# ----------------------------------------------- predictions ----------------------------------------------------------
def predict_single_audio(model, processor, audio_file_path):
    audio, rate = librosa.load(audio_file_path, sr=16000)
    inputs = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).to('cuda')
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    print(f"Transcription for {audio_file_path}: {transcription}")
    return transcription


def predict_for_competition_approach_1(model_silence_detection, model_speech_classification, processor, audio_dir):
    available_labels = get_available_labels(audio_dir)
    labels_to_classify = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    with open('./data/test/prediction_approach_1.csv', 'w') as f:
        f.write(f'fname,label\n')
        for audio_file in os.listdir("./data/test/audio"):
            audio, rate = librosa.load(os.path.join(audio_dir, audio_file), sr=16000)
            inputs = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000)
            with torch.no_grad():
                logits = model_silence_detection(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
            if len(transcription) > 0:
                with torch.no_grad():
                    logits = model_speech_classification(inputs.input_values).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.decode(predicted_ids[0])
                closest_label = min(available_labels, key=lambda l: Levenshtein.distance(l, transcription)).lower()
                if closest_label in labels_to_classify:
                    f.write(f'{audio_file},{closest_label}\n')
                else:
                    f.write(f'{audio_file},unknown\n')
            else:
                f.write(f'{audio_file},silence\n')


# ----------------------------------------------- callbacks ----------------------------------------------------------
class SaveMetricsCallback(TrainerCallback):
    def __init__(self, validation_dataset, name=''):
        self.validation_dataset = validation_dataset
        self.metrics_filename = f"data_for_vis/validation_metrics_{name}.csv"
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


class SaveMetricsCallbackSilence(TrainerCallback):
    def __init__(self, validation_dataset, name=''):
        self.validation_dataset = validation_dataset
        self.metrics_filename = f"data_for_vis/accuracy_silence_detection_{name}.csv"
        self.fieldnames = ["epoch", "accuracy", "accuracy_modification"]

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        accuracy = metrics["eval_accuracy"]
        accuracy_modification = metrics["eval_accuracy_modification"]
        epoch = state.epoch

        with open(self.metrics_filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if file.tell() == 0:  
                writer.writeheader()
            writer.writerow({"epoch": epoch, "accuracy": accuracy, "accuracy_modification": accuracy_modification})


# ----------------------------------------------- labels ----------------------------------------------------------
def get_available_labels(directory, include_silence=False):
    available_labels = [name.upper() for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    if not include_silence and 'SILENCE' in available_labels:
        available_labels.remove('SILENCE')
    return available_labels


def compute_metrics_with_closest_label(available_labels, predictions, labels):
    correct_count = 0
    total_count = len(predictions)
    total_cer = 0

    for prediction, label in zip(predictions, labels):
        closest_label = min(available_labels, key=lambda l: Levenshtein.distance(l, prediction))
        if closest_label == label:
            correct_count += 1
        else:
            if len(label) == 0:
                total_cer += len(closest_label)
            elif len(closest_label) == 0:
                total_cer += len(label)
            else:
                total_cer += cer(closest_label, label)

    accuracy = correct_count / total_count
    cer_score = total_cer / total_count
    return cer_score, accuracy


def calculate_cer_for_list(true_strings, predicted_strings):
    cer_total = 0
    for true, predicted in zip(true_strings, predicted_strings):
        if len(true) == 0:
            cer_total += len(predicted)
        elif len(predicted) == 0:
            cer_total += len(true)
        else:
            cer_total += cer(true, predicted)
    return cer_total / len(true_strings)