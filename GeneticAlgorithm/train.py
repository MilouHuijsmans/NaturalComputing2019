###code adapted from the github page: https://github.com/jliphard/DeepEvolve/blob/master/train.py

"""
Generic setup of the data sources and the model training. 
"""

#import modules

from __future__ import absolute_import, division, print_function
import numpy as np 
import pandas as pd 
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from keras.models         import Sequential
from keras.layers         import Dense, Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks      import EarlyStopping, Callback
from keras.layers         import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Activation
from keras                import backend as K

import logging

#set paths to images

infected_path = 'C:/Users/mikes/Desktop/milou/Natural Computing/cell_images/Parasitized/' 
uninfected_path = 'C:/Users/mikes/Desktop/milou/Natural Computing/cell_images/Uninfected/'

early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto' )

#Code adapted from the following kaggle kernel: 
#https://www.kaggle.com/kushal1996/detecting-malaria-cnn
def get_malaria_cnn():
    """Retrieve the malaria dataset and process the data."""
    
    infected = os.listdir(infected_path) 
    uninfected = os.listdir(uninfected_path)

    data = []
    labels = []
    
    for i in infected:
        try:

            image = cv2.imread(infected_path+i)
            image_array = Image.fromarray(image , 'RGB')
            resize_img = image_array.resize((100 , 100))
            data.append(np.array(resize_img))
            labels.append(1)

        except AttributeError:
            print('')
    
    for u in uninfected:
        try:

            image = cv2.imread(uninfected_path+u)
            image_array = Image.fromarray(image , 'RGB')
            resize_img = image_array.resize((100 , 100))
            data.append(np.array(resize_img))
            labels.append(0)

        except AttributeError:
            print('')
    
    cells = np.array(data)
    labels = np.array(labels)

    np.save('Cells' , cells)
    np.save('Labels' , labels)
    
    n = np.arange(cells.shape[0])
    np.random.shuffle(n)
    cells = cells[n]
    labels = labels[n]

    from sklearn.model_selection import train_test_split

    x_train , x_test , y_train , y_test = train_test_split(cells, labels, 
                                                test_size = 0.3,
                                                random_state = 111)
    
    nb_classes = 2 
    
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test,  nb_classes)

    input_shape = x_train.shape[1:]

    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    batch_size = 32
    epochs     = 3
    
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def compile_model_cnn(genome, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        genome (dict): the parameters of the genome

    Returns:
        a compiled network.

    """
    #get our network parameters
    dense = genome.geneparam['dense'] 
    activation = genome.geneparam['activation']
    optimizer  = genome.geneparam['optimizer' ]
    dropout  = genome.geneparam['dropout' ]
    conv_dropout= genome.geneparam['conv_dropout' ]

    logging.info("Architecture:%s,%s,%s,%s,%s" % (str(dense), str(dropout), str(conv_dropout), activation, optimizer))
    
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(conv_dropout))

    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(128, (5, 5), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_dropout))

    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(256, (5, 5), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(conv_dropout))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(dense))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def train_and_score(genome, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("Getting Keras datasets")

    nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_malaria_cnn()
  
    logging.info("Compiling Keras model")

    model = compile_model_cnn(genome, nb_classes, input_shape)

    history = LossHistory()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,  #using early stopping so no real limit - don't want to waste time on horrible architectures
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    K.clear_session()
    #we do not care about keeping any of this in memory - 
    #we just need to know the final scores and the architecture
    
    return score[1]  # 1 is accuracy. 0 is loss.