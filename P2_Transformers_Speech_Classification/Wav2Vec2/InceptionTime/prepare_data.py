from tensorflow.keras.utils import to_categorical
import os
import librosa 
import numpy as np
from sklearn.model_selection import train_test_split

def make_spec(file, file_dir):

    sig, _ = librosa.load(file_dir+file, sr=16000)
    
    if len(sig) < 16000: 
        sig = np.pad(sig, (0,16000-len(sig)), "linear_ramp")
   
    D = librosa.amplitude_to_db(librosa.stft(sig[:16000], 
                                             n_fft=512, 
                                             hop_length=128,
                                             center=False),
                               ref=np.max)
    S = librosa.feature.melspectrogram(S=D, n_mels=85).T
    
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


def read_classes(train_dir):
    classes = os.listdir(train_dir)
    if '.DS_Store' in classes:
        classes.remove(".DS_Store")
    NB_CLASSES = len(classes)
    def convert_list_dict(lst):
        res_dct = {i: val for i, val in enumerate(lst)}
        return res_dct
            
    classes_index = convert_list_dict(classes)
    return classes, classes_index,NB_CLASSES

def create_datastes_lists(train_dir = "./data/train/audio/", test_dir = None):
    classes = os.listdir(train_dir)
    if ".DS_Store" in classes:
        classes.remove(".DS_Store")
    training_list = []
    for cl in classes:
        for i, file in enumerate(os.listdir(train_dir+cl+"/")):
            training_list.append(cl+"/"+file)
    if test_dir:
        test_list = []
        for i, file in enumerate(os.listdir(test_dir)):
            test_list.append(file)
        return training_list, test_list
    return training_list



def create_prepare_data(train_dir):
    training_list = create_datastes_lists(train_dir =train_dir)
    classes, classes_index, NB_CLASSES = read_classes(train_dir)
    X_array, y_array = create_sets(file_list = training_list, classes = classes, file_dir=train_dir)
    X_train, X_val, y_train, y_val = train_test_split(X_array, y_array, train_size=0.7)
    X_train = X_train.reshape((-1, X_train.shape[1], X_train.shape[2]))
    X_val = X_val.reshape((-1, X_val.shape[1], X_val.shape[2]))


    y_train = to_categorical(y_train, num_classes=NB_CLASSES)
    y_val = to_categorical(y_val, num_classes=NB_CLASSES)
    return X_train, X_val, y_train, y_val
    