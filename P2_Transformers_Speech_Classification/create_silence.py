import os

import librosa
import numpy as np
import soundfile as sf


def split_arr(arr):
    return np.split(arr, np.arange(16000, len(arr), 16000))


def create_silence():
    train_dir = './data/train/audio/'
    for file in os.listdir("./data/train/audio/_background_noise_/"):
        if ".wav" in file:
            sig, sr = librosa.load("./data/train/audio/_background_noise_/" + file, sr=16000)
            sig_arr = split_arr(sig)
            if not os.path.exists(train_dir + "silence/"):
                os.makedirs(train_dir + "silence/")
            for ind, arr in enumerate(sig_arr):
                file_name = "frag%d" % ind + "_%s" % file
                sf.write(train_dir + "silence/" + file_name, arr, 16000)


def mix_sounds(sound1, sound2):
    min_len = min(len(sound1), len(sound2))
    sound1 = sound1[:min_len]
    sound2 = sound2[:min_len]
    mixed_sound = sound1 + sound2
    mixed_sound = mixed_sound / np.max(np.abs(mixed_sound))
    return mixed_sound


def create_mixed_silence():
    train_dir = './data/train/audio/'
    files = [file for file in os.listdir("./data/train/audio/_background_noise_/") if ".wav" in file]
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            file1 = files[i]
            file2 = files[j]
            sig1, sr1 = librosa.load("./data/train/audio/_background_noise_/" + file1, sr=16000)
            sig2, sr2 = librosa.load("./data/train/audio/_background_noise_/" + file2, sr=16000)
            mixed_sig = mix_sounds(sig1, sig2)
            sig_arr = split_arr(mixed_sig)
            if not os.path.exists(train_dir + "silence/"):
                os.makedirs(train_dir + "silence/")
            for ind, arr in enumerate(sig_arr):
                file_name = "mixed_frag%d" % ind + "_%s_%s" % (file1, file2)
                sf.write(train_dir + "silence/" + file_name, arr, 16000)


def create_mixed_silence_with_noise():
    train_dir = './data/train/audio/'
    files = [file for file in os.listdir("./data/train/audio/_background_noise_/") if ".wav" in file]
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            file1 = files[i]
            file2 = files[j]
            sig1, sr1 = librosa.load("./data/train/audio/_background_noise_/" + file1, sr=16000)
            sig2, sr2 = librosa.load("./data/train/audio/_background_noise_/" + file2, sr=16000)
            mixed_sig = mix_sounds(sig1, sig2)
            noise = np.random.normal(0, 0.05, mixed_sig.shape)
            mixed_sig = mixed_sig + noise
            sig_arr = split_arr(mixed_sig)
            if not os.path.exists(train_dir + "silence/"):
                os.makedirs(train_dir + "silence/")
            for ind, arr in enumerate(sig_arr):
                file_name = "mixed_noise_frag%d" % ind + "_%s_%s" % (file1, file2)
                sf.write(train_dir + "silence/" + file_name, arr, 16000)
