import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def plot_batch_training(dfs, trace_names, title):
    min_epoch = min(df['Epoch'].min() for df in dfs)
    max_epoch = max(df['Epoch'].max() for df in dfs)
    max_y = max([df[[' Accuracy', ' Accuracy_valid']].max().max() for df in dfs])
    min_y = min([df[[' Accuracy', ' Accuracy_valid']].min().min() for df in dfs])
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy Plot", "Validation Accuracy Plot"))
    for i, (df, trace_name) in enumerate(zip(dfs, trace_names)):
        color = colors[i]
        fig.add_trace(
            go.Scatter(x=df['Epoch'], y=df[' Accuracy'], mode='lines', name=trace_name, line=dict(color=color)), row=1,
            col=1)
        fig.add_trace(go.Scatter(x=df['Epoch'], y=df[' Accuracy_valid'], mode='lines', name=f'Gaussian Noise 0.{i + 1}',
                                 showlegend=False, line=dict(color=color)), row=1, col=2)

    fig.update_xaxes(title_text="Epoch", row=1, col=1, range=[min_epoch, max_epoch])
    fig.update_xaxes(title_text="Epoch", row=1, col=2, range=[min_epoch, max_epoch])
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, range=[min_y, max_y])
    fig.update_yaxes(title_text="Validation Accuracy", row=1, col=2, range=[min_y, max_y])
    fig.update_layout(title_text=title)
    fig.show()


def apply_transformation(image, transformation_layer):
    transformed_image = transformation_layer(image)
    return transformed_image


def plot_images(ds, transformation_layers, transformation_caption):
    sample_images, _ = next(iter(ds))
    image = sample_images[0]

    num_transformations = len(transformation_layers)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, num_transformations + 1, 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title('Original image')
    plt.axis('off')

    for i, transformation_layer in enumerate(transformation_layers):
        plt.subplot(1, num_transformations + 1, i + 2)
        transformed_image = apply_transformation(image, transformation_layer)
        plt.imshow(transformed_image.numpy().astype("uint8"))
        plt.title(f'{transformation_caption} image {i+1}')
        plt.axis('off')

    plt.show()


def mix_up(image1, image2, label1, label2, alpha=0.3):
    mixed_image = alpha * image1 + (1 - alpha) * image2
    mixed_label = alpha * label1 + (1 - alpha) * label2
    return mixed_image, mixed_label


def plot_mixup(ds):
    sample_images, sample_labels = next(iter(ds))

    image1, image2 = sample_images[0], sample_images[1]
    label1, label2 = sample_labels[0], sample_labels[1]
    mixed_image, mixed_label = mix_up(image1, image2, label1, label2)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image1.numpy().astype("uint8"))
    plt.title(f'Original image 1 (Label: {label1.numpy()})')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image2.numpy().astype("uint8"))
    plt.title(f'Original image 2 (Label: {label2.numpy()})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mixed_image.numpy().astype("uint8"))
    plt.title(f'Mixed image (Label: {mixed_label.numpy()})')
    plt.axis('off')

    plt.show()


def accuracy_boxplot(my_dir, name):
    df = merge_dfs(my_dir, name)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(x='Epoch', y=' Accuracy', data=df, ax=axs[0])
    axs[0].set_title('Accuracy for training set')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('accuracy')

    sns.boxplot(x='Epoch', y=' Accuracy_valid', data=df, ax=axs[1])
    axs[1].set_title('Accuracy for validation set')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('validation accuracy')

    plt.tight_layout()
    plt.show()


def accuracy_lineplot(my_dir, name):
    df = merge_dfs(my_dir, name)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    sns.lineplot(x='Epoch', y=' Accuracy', data=df, errorbar='sd', ax=axs[0])
    axs[0].set_title('Accuracy with standard deviation on training set')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('training accuracy')

    sns.lineplot(x='Epoch', y=' Accuracy_valid', data=df, errorbar='sd', ax=axs[1])
    axs[1].set_title('Accuracy with standard deviation on validation set')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('validation accuracy')

    plt.show()


def merge_dfs(my_dir, name):
    dfs = []
    for filename in os.listdir(my_dir):
        if filename.startswith(name):
            file_path = os.path.join(my_dir, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)

    return pd.concat(dfs)
