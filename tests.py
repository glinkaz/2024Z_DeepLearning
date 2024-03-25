from tensorflow import keras
import tensorflow as tf
import pickle


class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, file_path, val_file_path):
        super().__init__()
        self.file_path = file_path
        self.val_file_path = val_file_path
        self.batch_count = 0

    def on_train_begin(self, logs=None):
        with open(self.file_path, 'w') as file:
            file.write("Batch, Loss, Accuracy\n")
        with open(self.val_file_path, 'w') as file:
            file.write("Epoch, Loss, Accuracy, Loss_valid, Accuracy_valid\n")

    def on_batch_end(self, batch, logs=None):
        if self.batch_count % 10 == 0:
            with open(self.file_path, 'a') as file:
                file.write(f"{batch},{logs['loss']},{logs['accuracy']}\n")
        self.batch_count += 1

    def on_epoch_end(self, epoch, logs=None):
        with open(self.val_file_path, 'a') as file:
            file.write(f"{epoch},{logs['loss']},{logs['accuracy']}, {logs['val_loss']},{logs['val_accuracy']}\n")


def run_test(selected_model, train_ds, valid_ds, test_ds, model_name, test_name,
             preprocessing=tf.keras.layers.Identity(),
             optimizer=tf.keras.optimizers.Adam(), batch_size=16,
             dropout=tf.keras.layers.Identity(),
             save_predictions=False, n_epoch=30):

    model = tf.keras.Sequential([
        preprocessing,
        selected_model,
        dropout,
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_history_callback = LossHistory(
        file_path=f'../results/{model_name}_batch_{test_name}.csv',
        val_file_path=f'../results/{model_name}_epoch_{test_name}.csv'
    )

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=n_epoch,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3), loss_history_callback])
    _, test_acc = model.evaluate(test_ds)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))
    if save_predictions:
        predictions = model.predict(test_ds)
        with open(f'../predictions/pred_{model_name}_{test_name}.pickle', 'wb') as f:
            pickle.dump(predictions, f)
    keras.saving.save_model(model, f'../models/{model_name}_{test_name}.h5')



