import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

def vis_accuracy_cer(metrics_file1_path, metrics_file2_path):
    data1 = pd.read_csv(metrics_file1_path)
    data2 = pd.read_csv(metrics_file2_path)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(data1['epoch'], data1['accuracy'], label='lr=0.00005', color='mediumblue')
    axs[0].plot(data1['epoch'], data1['accuracy_closest'], label='lr=0.00005, with improvement', color='cornflowerblue')

    axs[0].plot(data2['epoch'], data2['accuracy'], label='lr=0.0001', color='darkorange')
    axs[0].plot(data2['epoch'], data2['accuracy_closest'], label='lr=0.0001, with improvement', color='gold')

    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('accuracy')
    axs[0].set_title('Validation data accuracy across epochs')
    axs[0].legend()

    axs[1].plot(data1['epoch'], data1['cer'], label='lr=0.00005', color='mediumblue')
    axs[1].plot(data1['epoch'], data1['cer_closest'], label='lr=0.00005, with improvement', color='cornflowerblue')

    axs[1].plot(data2['epoch'], data2['cer'], label='lr=0.0001', color='darkorange')
    axs[1].plot(data2['epoch'], data2['cer_closest'], label='lr=0.0001, with improvement', color='gold')

    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('CER')
    axs[1].set_title('Validation data CER across epochs')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def vis_accuracy_cer_one_file(metrics_file_path):
    data1 = pd.read_csv(metrics_file_path)
    plt.figure(figsize=(8, 6))
    plt.plot(data1['epoch'], data1['accuracy'], label='model without modification', color='mediumblue')
    plt.plot(data1['epoch'], data1['accuracy_closest'], label='model with modification', color='cornflowerblue')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Validation data accuracy across epochs')
    plt.tight_layout()
    plt.legend()
    plt.show()


def vis_silence_detection_accuracy(metrics_file_path):
    data = pd.read_csv(metrics_file_path)

    plt.figure(figsize=(8, 6))
    plt.plot(data['epoch'], data['accuracy'], label='standard classification', color='darkblue')
    plt.plot(data['epoch'], data['accuracy_modification'], label='classification with modification',
             color='cornflowerblue')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Validation data accuracy across epochs')
    plt.legend()
    plt.show()

def show_confusion_matrix(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='g', cmap='viridis', norm=colors.LogNorm())
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()