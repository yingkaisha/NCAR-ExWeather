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

from sklearn.metrics import classification_report, auc, roc_curve
from sklearn.metrics import confusion_matrix

def verif_metric(VALID_target, Y_pred, thres=0.5):

    tn, fp, fn, tp = confusion_matrix(VALID_target.ravel(), Y_pred.ravel()>thres).ravel()

    CSI = tp/(tp+fn+fp)
    CSI_metric = 1 - CSI
    
    POFD = fp/(tn+fp)
    
    fpr, tpr, thresholds = roc_curve(VALID_target.ravel(), Y_pred.ravel())
    AUC = auc(fpr, tpr)
    AUC_metric = 1 - AUC
    print('{} {} {}'.format(CSI, POFD, AUC))
    metric = 0.2*POFD + 0.7*CSI_metric + 0.1*AUC_metric


    return metric

# ========== training/validation split ========== #

filename_aug = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_aug/*.npy"))
filename_full = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch/*.npy"))

cut_train_aug = 136000
cut_train_full = 2848299

filename_train_aug = filename_aug[:cut_train_aug]
filename_train_full = filename_full[:cut_train_full]

factor = 20

L_valid_aug = int(5*factor)
L_valid_full = int(3000*factor)

filename_valid_aug = filename_aug[cut_train_aug:]
filename_valid_full = filename_full[cut_train_full:]

shuffle(filename_valid_aug)
shuffle(filename_valid_full)

filename_valid_aug = filename_aug[-L_valid_aug:]
filename_valid_full = filename_full[-L_valid_full:]

# ========== Validation set ========== #

ind_pick_from_batch = [11, 12, 2, 17, 13, 1]

L_vars = len(ind_pick_from_batch)

grid_shape = (128, 128)

L_valid = L_valid_aug+L_valid_full

VALID_input = np.empty((L_valid,)+grid_shape+(L_vars,))
VALID_target = np.empty(L_valid)

for i, filename in enumerate(filename_valid_aug+filename_valid_full):
    data = np.load(filename)
    
    for c, v in enumerate(ind_pick_from_batch):
    
        VALID_input[i, ..., c] = data[..., v]
        
    if 'pos' in filename:
        VALID_target[i] = True
    elif 'neg' in filename:
        VALID_target[i] = False
    else:
        aergheagtha

# ========== Model ========== #

# ---------- Layers ---------- #

IN = tf.keras.Input((128, 128, 6))

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

OUT = D

model = keras.models.Model(inputs=[IN,], outputs=[OUT,])

W_new = model.get_weights()

# ---------- Weights ---------- #

temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

key = 'BIG6'

model_name = '{}_tornado'.format(key)
model_path = temp_dir+model_name

#W_new = k_utils.dummy_loader(model_path)

W_old = k_utils.dummy_loader(temp_dir+'VGG_X_pp20_tune2')

for l in range(len(W_old)):
    if W_old[l].shape == W_new[l].shape:
        W_new[l] = W_old[l]

# ---------- Compile ---------- #

model.set_weights(W_new)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=1e-5))

# ========== Initial record ========== #

#Y_pred = model.predict([VALID_input,])
record = 0.7 #verif_metric(VALID_target, Y_pred)
print('Initial record {}'.format(record))

# ========== Training hyper parameters ========== #

tol = 0
min_del = 0
max_tol = 500 # early stopping with patience

epochs = 500
L_train = 64
batch_size = 100
batch_size_half = 50

valid_size = 1

X_batch = np.empty((batch_size, 128, 128, L_vars))
Y_batch = np.empty((batch_size, 1))
X_batch[...] = np.nan
Y_batch[...] = np.nan

# ========== Training loop ========== #

L_full = len(filename_train_full)
L_aug = len(filename_train_aug)

for i in range(epochs):
    
    if i <= 10:
        batch_size_full = 70
    if i > 10 and i <= 35:
        batch_size_full = 85
    if i > 35:
        batch_size_full = 95
    
    batch_size_aug = batch_size - batch_size_full
    
    #print('epoch = {}'.format(i))
    start_time = time.time()
    
    # loop of batch
    for j in range(L_train):
        
        ind_full = du.shuffle_ind(L_full)
        ind_aug = du.shuffle_ind(L_aug)
        
        file_pick_full = []
        for ind_temp in ind_full[:batch_size_full]:
            file_pick_full.append(filename_train_full[ind_temp])

        file_pick_aug = []
        for ind_temp in ind_aug[:batch_size_aug]:
            file_pick_aug.append(filename_train_aug[ind_temp])
        
        file_pick = file_pick_full + file_pick_aug
        
        for k in range(batch_size):
            
            data = np.load(file_pick[k])
            
            for c, v in enumerate(ind_pick_from_batch):
                
                X_batch[k, ..., c] = data[..., v]
            
            if 'pos' in file_pick[k]:
                Y_batch[k, :] = np.random.uniform(0.95, 0.99)
            elif 'neg' in file_pick[k]:
                Y_batch[k, :] = np.random.uniform(0.01, 0.05)
            else:
                werhgaer
        
        # # add noise within sparse inputs
        # for v in flag_sparse:
        #     X_batch[..., v] += np.random.uniform(0, 0.01, size=(batch_size, 128, 128))

        # shuffle indices
        ind_ = du.shuffle_ind(batch_size)
        X_batch = X_batch[ind_, ...]
        Y_batch = Y_batch[ind_, :]
        
        if np.sum(np.isnan(X_batch)) > 0:
            asfeargagqarew
        
        # train on batch
        model.train_on_batch([X_batch,], [Y_batch,]);
    
    # epoch end operations
    Y_pred = model.predict([VALID_input,])
    record_temp = verif_metric(VALID_target, Y_pred, thres=0.5)
    
    if (record - record_temp > min_del) and (np.max(Y_pred) > 0.6):
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        tol = 0
        #print('tol: {}'.format(tol))
        # save
        print('save to: {}'.format(model_path))
        model.save(model_path)
    else:
        print('Validation loss {} NOT improved'.format(record_temp))
        tol += 1
        #print('tol: {}'.format(tol))
        if tol >= max_tol:
            print('Early stopping')
            sys.exit();
        else:
            #print('Pass to the next epoch')
            continue;
    print("--- %s seconds ---" % (time.time() - start_time))
