import numpy as np
import keras
from keras.layers import Conv1D, BatchNormalization, Activation, Input, Dense, Bidirectional, LSTM, Dropout, TimeDistributed, Lambda
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from livelossplot import PlotLossesKeras
from tensorflow.keras.metrics import Recall, Precision
import keras.backend as K
import os
# edit_distance = keras_nlp.metrics.EditDistance()
class RNN_CTC():

    def __init__(self, input_shape, nb_classes, weights_directory='./', nb_epochs=100, batch_size=64):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.weights_directory = weights_directory
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.m = None
        self.tm = None    
        
        self.build()
        
    def ctc_layer_func(self, args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    def ctc_loss(self, y_true, y_pred):
        return y_pred
        
    def build(self, conv_filters=128, conv_size=12, conv_strides=4, activation="relu", rnn_layers=2, lstm_units=64, drop_out=0.6):   
        
        inputs = Input(shape=self.input_shape, name='input')
        x = Conv1D(conv_filters, 
                   conv_size, 
                   strides = conv_strides, 
                   name = 'conv1d')(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        for _ in range(rnn_layers):          
            x = Bidirectional(LSTM(lstm_units, 
                                   return_sequences = True))(x)
            x = Dropout(drop_out)(x)
            x = BatchNormalization()(x)
        outputs = TimeDistributed(Dense(self.nb_classes, activation="softmax"))(x)
        
        labels = Input(name="the_labels", shape=[None,], dtype="int32")
        input_length = Input(name="input_length", shape=[1], dtype="int32")
        label_length = Input(name="label_length", shape=[1], dtype="int32")
        
        ctc_layer = Lambda(self.ctc_layer_func, output_shape=(1,), name="ctc")([outputs, labels, input_length, label_length])
        self.tm = Model(inputs=inputs, outputs=outputs)
        self.m = Model(inputs=[inputs,labels,input_length,label_length],
                       outputs=ctc_layer)            
        
        self.m.compile(loss=self.ctc_loss, 
                       optimizer=Adam(),
                    #    metrics=[ tf.edit_distance]
                       )
        
        self.tm.compile(loss=self.ctc_loss, 
                        optimizer=Adam())
              
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                      factor=0.5, 
                                      patience=int(self.nb_epochs/20),
                                      min_lr=0.0001)
        
        file_path = os.path.join(self.weights_directory,"ctc_best_weights.h5")
        model_checkpoint = ModelCheckpoint(filepath=file_path, 
                                           monitor='val_loss',
                                           mode="max",
                                           save_best_only=True)
        
        early_stopping = EarlyStopping(monitor="val_loss", 
                                       mode="max", 
                                       verbose=1, 
                                       patience=int(self.nb_epochs/10))
        plotlosses = PlotLossesKeras()
        self.callbacks = [reduce_lr, model_checkpoint, early_stopping, plotlosses]
     
        print(self.m.summary())
        print(self.tm.summary())
        
        return self.m, self.tm
    
    
    def fit(self, X_train, train_labels, train_input_length, train_label_length, y_train,
                  X_val, val_labels, val_input_length, val_label_length, y_val):       

        hist = self.m.fit([np.squeeze(X_train), 
                            train_labels, 
                            train_input_length, 
                            train_label_length], 
                       np.zeros([len(y_train)]), 
                       batch_size = self.batch_size, 
                       epochs = self.nb_epochs, 
                       validation_data = ([np.squeeze(X_val), 
                                           val_labels, 
                                           val_input_length, 
                                           val_label_length],
                                          np.zeros([len(y_val)])), 
                       callbacks  = self.callbacks, 
                       verbose = 1, 
                       shuffle = True,
                    )
        
        keras.backend.clear_session()
        return hist
    
    def predict(self, dataset, index_map):
        k_ctc_out = K.ctc_decode(self.tm.predict(np.squeeze(dataset), 
                                                verbose = 1), 
                             np.array([28 for _ in dataset]))
        decoded_out = K.eval(k_ctc_out[0][0])
        str_decoded_out = []
        for i, _ in enumerate(decoded_out):     
            str_decoded_out.append("".join([index_map[c] for c in decoded_out[i] if not c == -1]))

        return str_decoded_out