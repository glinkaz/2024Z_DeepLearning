from tensorflow.keras.utils import to_categorical
import os
import librosa 
import numpy as np
from sklearn.model_selection import train_test_split
from jiwer import cer
import Levenshtein


def make_spec(file, file_dir, flip=False, ps=False, st = 4):

    sig, sr = librosa.load(file_dir+file, sr=16000)
    
    if len(sig) < 16000: 
        sig = np.pad(sig, (0,16000-len(sig)), "linear_ramp")
   
    D = librosa.amplitude_to_db(librosa.stft(sig[:16000], 
                                             n_fft=512, 
                                             hop_length=128,
                                             center=False),
                               ref=np.max)
    S = librosa.feature.melspectrogram(S=D, n_mels=85).T
    
    if flip:
        S = np.flipud(S)
    
    return S.astype(np.float32)

def create_sets(file_list, classes, file_dir):

    X_array = np.zeros([len(file_list), 122, 85])
    y_array = np.zeros([len(file_list)])
    for ind, file in enumerate(file_list):
        if  '.wav' not in file:
           continue
        if ind%2000 == 0:
            print(ind, file)
  
        X_array[ind] = make_spec(file, file_dir = file_dir)
        y_array[ind] = classes.index(file.rsplit('/')[0])
        
    return X_array, y_array

def convert_list_dict(lst):
    res_dct = {i: val for i, val in enumerate(lst)}
    return res_dct

def text_to_int(text, char_map):

    int_seq = []
    if text == 'silence':
        for r in range(8):
            int_seq.append(27)
    else:
        for c in text:
            ch = char_map[c]
            int_seq.append(ch)
    return int_seq

def get_intseq(trans, char_map, max_len = 8):
  
    t = text_to_int(trans, char_map)
    while (len(t) < max_len):
        t.append(27)
    return t

def get_ctc_params(y, classes_list,char_map, len_char_map = 28):

    labels = np.array([get_intseq(classes_list[y[l]], char_map) for l, _ in enumerate(y)])
    input_length = np.array([len_char_map for _ in y])
    label_length = np.array([8 for _ in y])
    return labels, input_length, label_length


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


def classify_as_unknwn(classes):
    return [i if i in ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence'] else 'unknown' for i in classes]