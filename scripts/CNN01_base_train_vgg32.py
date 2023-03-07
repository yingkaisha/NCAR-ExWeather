# general tools
import os
import sys
from glob import glob

# data tools
import time
import h5py
import random
import numpy as np
from random import shuffle

from datetime import datetime, timedelta
import re

# deep learning tools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import utils
from tensorflow.keras import Model

tf.config.run_functions_eagerly(True)

from keras_unet_collection import utils as k_utils

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

# ==================== #
weights_round = 2
save_round = 3
seeds = 777
# ==================== #

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
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

def create_model_head():
    
    IN_vec = keras.Input((128,))    
    X = IN_vec
    #
    X = keras.layers.Dense(64)(X)
    X = keras.layers.Activation("relu")(X)
    X = keras.layers.BatchNormalization()(X)
    
    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=IN_vec, outputs=OUT)
    
    return model

def verif_metric(VALID_target, Y_pred):
    
    BS = np.mean((VALID_target.ravel() - Y_pred.ravel())**2)
    
    print('{}'.format(BS))
    metric = BS

    return metric

def name_extract(filenames):
    
    date_base = datetime(2020, 7, 14)
    date_base2 = datetime(2021, 1, 1)
    
    filename_train = []
    filename_valid = []
    
    base_v3_s = datetime(2018, 7, 15)
    base_v3_e = datetime(2020, 12, 2)

    base_v4_s = datetime(2020, 12, 3)
    base_v4_e = datetime(2022, 7, 15)

    base_ref = datetime(2010, 1, 1)
    
    date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
    date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+365+30)]
    
    base_ref = datetime(2019, 10, 1)
    date_list_v4x = [base_ref + timedelta(days=day) for day in range(429)]
    
    for i, name in enumerate(filenames):
        
        if 'v4x' in name:
            date_list = date_list_v4x
        elif 'v4' in name:
            date_list = date_list_v4
        else:
            date_list = date_list_v3
        
        nums = re.findall(r'\d+', name)
        day = int(nums[-4])
        day = date_list[day]
        
        if (day - date_base).days < 0:
            filename_train.append(name)
            
        else:
            if (day - date_base2).days < 0:
                filename_valid.append(name)
        
    return filename_train, filename_valid


ind_pick_from_batch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
L_vars = len(ind_pick_from_batch)

vers = ['v3', 'v4x', 'v4']
leads = [2, 3, 4, 5, 6, 20, 21, 22, 23]
filenames_pos = {}
filenames_neg = {}

for ver in vers:
    for lead in leads:
        if ver == 'v3' and lead < 23:
            path_ = '/glade/scratch/ksha/DATA/NCAR_batch_v3/'
        elif ver == 'v3' and lead == 23:
            path_ = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/'
        elif ver == 'v4':
            path_ = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'
        else:
            path_ = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'
            
        filenames_pos['{}_lead{}'.format(ver, lead)] = sorted(glob("{}*pos*lead{}.npy".format(path_, lead)))
        filenames_neg['{}_lead{}'.format(ver, lead)] = sorted(glob("{}*neg_neg_neg*lead{}.npy".format(path_, lead)))
        
        print('{}, lead{}, pos: {}, neg: {}'.format(ver, lead, 
                                                    len(filenames_pos['{}_lead{}'.format(ver, lead)]), 
                                                    len(filenames_neg['{}_lead{}'.format(ver, lead)])))
        
filenames_pos_train = {}
filenames_neg_train = {}

filenames_pos_valid = {}
filenames_neg_valid = {}

for ver in vers:
    for lead in leads:
        temp_namelist_pos = filenames_pos['{}_lead{}'.format(ver, lead)]
        temp_namelist_neg = filenames_neg['{}_lead{}'.format(ver, lead)]
        
        pos_train, pos_valid = name_extract(temp_namelist_pos)
        neg_train, neg_valid = name_extract(temp_namelist_neg)
        
        print('pos train: {} pos valid: {} neg train: {} neg valid {}'.format(len(pos_train), 
                                                                              len(pos_valid), 
                                                                              len(neg_train), 
                                                                              len(neg_valid)))
        
        filenames_pos_train['{}_lead{}'.format(ver, lead)] = pos_train
        filenames_neg_train['{}_lead{}'.format(ver, lead)] = neg_train
        
        filenames_pos_valid['{}_lead{}'.format(ver, lead)] = pos_valid
        filenames_neg_valid['{}_lead{}'.format(ver, lead)] = neg_valid
        
pos_train_all = []
neg_train_all = []
pos_valid_all = []
neg_valid_all = []

for ver in vers:
    for lead in leads:
        pos_train_all += filenames_pos_train['{}_lead{}'.format(ver, lead)]
        neg_train_all += filenames_neg_train['{}_lead{}'.format(ver, lead)]
        pos_valid_all += filenames_pos_valid['{}_lead{}'.format(ver, lead)]
        neg_valid_all += filenames_neg_valid['{}_lead{}'.format(ver, lead)]
        
save_dir = '/glade/work/ksha/NCAR/'

with h5py.File(save_dir+'CNN_Validation.hdf', 'r') as h5io:
    VALID_input_32 = h5io['VALID_input_64'][:, 16:-16, 16:-16, ...]
    VALID_target = h5io['VALID_target'][...]
    
flag_train = 'base'

if flag_train == 'head':
    flag_weights = 'base'
else:
    flag_weights = 'head'
    
model_head = create_model_head()
model_base = create_model(input_shape=(32, 32, 15))

IN = layers.Input(shape=(32, 32, 15))

VEC = model_base(IN)
OUT = model_head(VEC)

model_final = Model(inputs=IN, outputs=OUT)
model_final.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer=keras.optimizers.Adam(lr=1e-4))
W_old = k_utils.dummy_loader('/glade/work/ksha/NCAR/Keras_models/RE2_vgg_base{}/'.format(weights_round))
model_final.set_weights(W_old)

Y_pred = model_final.predict([VALID_input_32])
record_temp = verif_metric(VALID_target, Y_pred)

min_del = 0
max_tol = 100 # early stopping with patience

epochs = 500
batch_size = 200
L_train = 64 #int(len(TRAIN_Y_pick) / batch_size)

X_batch_32 = np.empty((batch_size, 32, 32, L_vars))
Y_batch = np.empty((batch_size, 1))

X_batch_32[...] = np.nan
Y_batch[...] = np.nan

temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

# =========== Model Section ========== #
key = 'RE2_vgg_{}{}'.format(flag_train, save_round)
model_name = '{}'.format(key)
model_path = temp_dir+model_name

tol = 0

filename_pos_train = pos_train_all
filename_neg_train = neg_train_all
# ========== Training loop ========== #
L_pos = len(filename_pos_train)
L_neg = len(filename_neg_train)

record = record_temp #0.01840167896363949 #record_temp
print("Initial record: {}".format(record))

set_seeds(seeds)
    
for i in range(epochs):
    start_time = time.time()

    # loop of batch
    for j in range(L_train):
        if flag_train == 'base':
            N_pos = 20
        else:
            N_pos = 20
            
        N_neg = batch_size - N_pos

        ind_neg = du.shuffle_ind(L_neg)
        ind_pos = du.shuffle_ind(L_pos)

        file_pick_neg = []
        for ind_temp in ind_neg[:N_neg]:
            file_pick_neg.append(filename_neg_train[ind_temp])

        file_pick_pos = []
        for ind_temp in ind_pos[:N_pos]:
            file_pick_pos.append(filename_pos_train[ind_temp])

        file_pick = file_pick_neg + file_pick_pos

        if len(file_pick) != batch_size:
            sregwet

        for k in range(batch_size):
            data = np.load(file_pick[k])

            for l, c in enumerate(ind_pick_from_batch):
                temp = data[:, 16:-16, 16:-16, c] 
                X_batch_32[k, ..., l] = temp

            if 'pos' in file_pick[k]:
                Y_batch[k, :] = 1.0 #np.random.uniform(0.9, 0.99)
            elif 'neg_neg_neg' in file_pick[k]:
                Y_batch[k, :] = 0.0 #np.random.uniform(0.01, 0.05)
            else:
                werhgaer

        ind_ = du.shuffle_ind(batch_size)
        X_batch_32 = X_batch_32[ind_, ...]
        Y_batch = Y_batch[ind_, :]

        # train on batch
        model_final.train_on_batch(X_batch_32, Y_batch);

    # epoch end operations
    Y_pred = model_final.predict([VALID_input_32])
    # Y_pred[Y_pred<0] = 0
    # Y_pred[Y_pred>1] = 1

    record_temp = verif_metric(VALID_target, Y_pred)

    # if i % 10 == 0:
    #     model.save(model_path_backup)

    if (record - record_temp > min_del):
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        tol = 0
        #print('tol: {}'.format(tol))
        # save
        print('save to: {}'.format(model_path))
        model_final.save(model_path)
    else:
        print('Validation loss {} NOT improved'.format(record_temp))
        if record_temp >= 2.0:
            print('Early stopping')
            break;
        else:
            tol += 1
            if tol >= max_tol:
                print('Early stopping')
                break;
            else:
                continue;
    print("--- %s seconds ---" % (time.time() - start_time))
