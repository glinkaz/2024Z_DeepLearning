from typing import Tuple
from tensorflow import keras


def load_set(path: str, label_mode: str = 'categorical', batch_size: int = 16,
             image_size: Tuple[int, int] = (32, 32)):
    return keras.preprocessing.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode=label_mode,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1111,
    )


def load_data(train_path: str, valid_path: str, test_path: str, label_mode: str = 'categorical', batch_size: int = 16,
             image_size: Tuple[int, int] = (32, 32)):
    train_ds = load_set(path=train_path, label_mode=label_mode, batch_size=batch_size, image_size=image_size)
    valid_ds = load_set(valid_path, label_mode=label_mode, batch_size=batch_size, image_size=image_size)
    test_ds = load_set(test_path, label_mode=label_mode, batch_size=batch_size, image_size=image_size)
    return train_ds, valid_ds, test_ds