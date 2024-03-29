
# general tools
import sys
from glob import glob

# data tools
import time
import h5py
import numpy as np

# deep learning tools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import Model
from keras_unet_collection import utils as k_utils

#tf.config.run_functions_eagerly(True)

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('part', help='part')
parser.add_argument('lead', help='lead')
parser.add_argument('model', help='model')
parser.add_argument('prefix', help='prefix')
parser.add_argument('ver', help='ver')
args = vars(parser.parse_args())

# =============== #
part = int(args['part'])
lead = int(args['lead'])

model_name = args['model']
prefix = args['prefix']
ver = args['ver']

gap = 200000

ind_pick_from_batch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
L_vars = len(ind_pick_from_batch)

# Collection batch names from scratch and campaign
names = glob("/glade/scratch/ksha/DATA/NCAR_batch_{}/TRAIN*lead{}.npy".format(ver, lead)) + glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch_{}/TRAIN*lead{}.npy".format(ver, lead))

filename_train = sorted(names)

# divide batch files into three parts
if part == 0:
    filename_train = filename_train[:gap]
elif part == 1:
    filename_train = filename_train[gap:2*gap]
else:
    filename_train = filename_train[2*gap:]

L_train = len(filename_train)
L_var = L_vars

# Combine individual batch files to a large array
TEST_input_64 = np.empty((L_train, 32, 32, L_var))
TEST_target = np.ones(L_train)

for i, name in enumerate(filename_train):
    data = np.load(name)
    for k, c in enumerate(ind_pick_from_batch):
        
        TEST_input_64[i, ..., k] = data[:, 16:-16, 16:-16, c]

        if 'pos' in name:
            TEST_target[i] = 1.0
        else:
            TEST_target[i] = 0.0
            
def create_model(input_shape=(32, 32, 15)):

    channels = [32, 64, 96, 128]

    Input_shape=(32, 32, 15)
    IN = layers.Input(shape=Input_shape)

    X = IN

    X = keras.layers.Conv2D(channels[0], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[0], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)
    # pooling
    X = keras.layers.Conv2D(channels[1], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

    X = keras.layers.Conv2D(channels[1], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[1], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    # pooling
    X = keras.layers.Conv2D(channels[2], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

    X = keras.layers.Conv2D(channels[2], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[2], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    # pooling
    X = keras.layers.Conv2D(channels[3], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

    X = keras.layers.Conv2D(channels[3], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[3], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    V1 = X
    OUT = keras.layers.GlobalMaxPooling2D()(V1)
    model = Model(inputs=IN, outputs=OUT)
    
    return model

# Crerate model
model = create_model(input_shape=(32, 32, 15))

# get current weights
W_new = model.get_weights()

# get stored weights
print('/glade/work/ksha/NCAR/Keras_models/{}/'.format(model_name))
W_old = k_utils.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}/'.format(model_name))

# update stored weights to new weights
for i in range(len(W_new)):
    if W_new[i].shape == W_old[i].shape:
        W_new[i] = W_old[i]
    else:
        # the size of the weights always match, this will never happen
        ewraewthws

# dump new weights to the model
model.set_weights(W_new)

# compile just in case
model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.SGD(lr=0))

# predict feature vectors
Y_vector = model.predict([TEST_input_64,])

# Save as numpy file
save_dict = {}
save_dict['y_true'] = TEST_target
save_dict['y_vector'] = Y_vector
save_name = "/glade/work/ksha/NCAR/TRAIN_{}_32_lead{}_part{}_{}.npy".format(ver, lead, part, prefix)
print(save_name); np.save(save_name, save_dict)


