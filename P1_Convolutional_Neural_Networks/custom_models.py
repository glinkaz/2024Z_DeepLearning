import keras
from tensorflow.python.keras.regularizers import l2, l1
import tensorflow as tf


def custom_model1():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
    ])
    return model


def custom_model2(weight_decay=0.0001):
    model = keras.Sequential([
        keras.layers.Rescaling(1. / 255),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                            kernel_regularizer=l2(weight_decay),
                            input_shape=(32, 32)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                            kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(rate=0.2),

        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                            kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                            kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(rate=0.3),

        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                            kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                            kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                            kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                            kernel_regularizer=l2(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.5),

        keras.layers.Flatten(),
    ])
    return model


def get_efficientnet():
    efficientNetV2B0 = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        input_shape=(224, 224, 3),
        include_top=True,
        weights='imagenet'
    )
    efficientNetV2B0.trainable = False
    return efficientNetV2B0


def custom_model1_regularized(weight_decay = 0.0001):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', padding='same',kernel_regularizer=l1(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l1(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l1(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l1(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l1(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l1(weight_decay)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
    ])
    return model
