# general tools
import sys
from glob import glob

# data tools
import time
import h5py
import random
import numpy as np
from random import shuffle

# deep learning tools
import tensorflow as tf
from tensorflow import keras

from keras_unet_collection import models as k_models
from keras_unet_collection import utils as k_utils
from keras_unet_collection import layer_utils as k_layers
from keras_unet_collection.activations import GELU

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du
import graph_utils as gu

from sklearn.metrics import classification_report, auc, roc_curve
from sklearn.metrics import confusion_matrix

def verif_metric(VALID_target, Y_pred, thres=0.5):

    tn, fp, fn, tp = confusion_matrix(VALID_target.ravel(), Y_pred.ravel()>thres).ravel()

    CSI = tp/(tp+fn+fp)
    
    POFD = fp/(tn+fp)
    
    fpr, tpr, thresholds = roc_curve(VALID_target.ravel(), Y_pred.ravel())
    AUC = auc(fpr, tpr)


    return CSI, POFD, AUC

filename_aug = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_aug/*.npy"))
filename_full = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch/*.npy"))

cut_train_aug = 136000
cut_train_full = 2848299

filename_train_aug = filename_aug[:cut_train_aug]
filename_train_full = filename_full[:cut_train_full]

L_valid_aug = 2000
L_valid_full = 6000

filename_valid_aug = filename_aug[-L_valid_aug:]
filename_valid_full = filename_full[-L_valid_full:]

L_vars = 19
grid_shape = (128, 128)

L_valid = L_valid_aug+L_valid_full

VALID_input = np.empty((L_valid,)+grid_shape+(L_vars,))
VALID_target = np.empty(L_valid)
for i, filename in enumerate(filename_valid_aug+filename_valid_full):
    data = np.load(filename)
    VALID_input[i, ...] = data[..., :L_vars]
    if 'pos' in filename:
        VALID_target[i] = True
    elif 'neg' in filename:
        VALID_target[i] = False
    else:
        aergheagtha

IN = tf.keras.Input((128, 128, 19))

X = IN

X = k_layers.CONV_stack(X, 32, kernel_size=3, stack_num=2, dilation_rate=1, activation='GELU', batch_norm=True, name='conv_stack1')
X = tf.keras.layers.Conv2D(32, kernel_size=2, strides=(2, 2), padding='valid', use_bias=True, name='stride_conv1')(X)

X = k_layers.CONV_stack(X, 64, kernel_size=3, stack_num=2, dilation_rate=1, activation='GELU', batch_norm=True, name='conv_stack2')
X = tf.keras.layers.Conv2D(64, kernel_size=2, strides=(2, 2), padding='valid', use_bias=True, name='stride_conv2')(X)

X = k_layers.CONV_stack(X, 128, kernel_size=3, stack_num=2, dilation_rate=1, activation='GELU', batch_norm=True, name='conv_stack3')
X = tf.keras.layers.Conv2D(128, kernel_size=2, strides=(2, 2), padding='valid', use_bias=True, name='stride_conv3')(X)

X = k_layers.CONV_stack(X, 256, kernel_size=3, stack_num=2, dilation_rate=1, activation='GELU', batch_norm=True, name='conv_stack4')
X = tf.keras.layers.Conv2D(256, kernel_size=2, strides=(2, 2), padding='valid', use_bias=True, name='stride_conv4')(X)

X = k_layers.CONV_stack(X, 512, kernel_size=3, stack_num=2, dilation_rate=1, activation='GELU', batch_norm=True, name='conv_stack5')
X = tf.keras.layers.Conv2D(512, kernel_size=2, strides=(2, 2), padding='valid', use_bias=True, name='stride_conv5')(X)

D = tf.keras.layers.Flatten()(X)

D = tf.keras.layers.Dense(512, use_bias=False, name='dense1')(D)
D = tf.keras.layers.BatchNormalization(axis=-1, name='dense_bn1')(D)
D = GELU()(D)

D = tf.keras.layers.Dense(128, use_bias=False, name='dense2')(D)
D = tf.keras.layers.BatchNormalization(axis=-1, name='dense_bn2')(D)
D = GELU()(D)

D = tf.keras.layers.Dense(1, activation='sigmoid', name='head')(D)
#D = tf.keras.layers.Softmax()(D)

OUT = D

model = keras.models.Model(inputs=[IN,], outputs=[OUT,])
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=2e-5))

temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

key = 'VGG_X'

model_name = '{}_pp20_tune2'.format(key)
model_path = temp_dir+model_name

W = k_utils.dummy_loader(model_path)
model.set_weights(W)

rounds = 2
N_var = 19
AUC_drop = np.empty((N_var, 3, 2))

for r in range(rounds):
    for n in range(N_var):
        start_time = time.time()
        print('shuffling var {}'.format(n))

        ind_ = du.shuffle_ind(L_valid)
        VALID_input_shuffle = np.copy(VALID_input)
        VALID_input_shuffle[..., n] = VALID_input_shuffle[ind_, ..., n]

        Y_pred = model.predict([VALID_input_shuffle,])
        AUC_drop[n, :, r] = verif_metric(VALID_target, Y_pred, thres=0.5)
        print("--- %s seconds ---" % (time.time() - start_time))

np.save(temp_dir+model_name+'_FI.npy', AUC_drop)

