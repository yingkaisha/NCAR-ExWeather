'''
Training VGG-based CNN to produce feature vectors.
(Revamped)
'''

## general tools
import os
import sys
from glob import glob

# data tools
import re
import time
import h5py
import random
import numpy as np
from random import shuffle
from datetime import datetime, timedelta

#tf.config.run_functions_eagerly(True)

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

# ==================== #
weights_round = 0
save_round = 1
seeds = 777
model_prefix_load = 'RE2_vgg_base{}'.format(weights_round) #False
model_prefix_save = 'RE2_vgg_base{}'.format(save_round)
N_vars = L_vars = 15
# ==================== #

# ----------------------------------------------------- #
# Collect pos and neg batch filenames

vers = ['v3', 'v4x', 'v4'] # HRRR v4, v4x, v4
leads = [2, 3, 4, 5, 6, 20, 21, 22, 23]

filenames_pos = {}
filenames_neg = {}

# Identify and separate pos / neg batch files
for ver in vers:
    for lead in leads:
        if ver == 'v3':
            path_ = path_batch_v3
        elif ver == 'v4':
            path_ = path_batch_v4
        else:
            path_ = path_batch_v4x
            
        filenames_pos['{}_lead{}'.format(ver, lead)] = sorted(glob("{}*pos*lead{}.npy".format(path_, lead)))
        filenames_neg['{}_lead{}'.format(ver, lead)] = sorted(glob("{}*neg_neg_neg*lead{}.npy".format(path_, lead)))
        
        print('{}, lead{}, pos: {}, neg: {}'.format(ver, lead, len(filenames_pos['{}_lead{}'.format(ver, lead)]), 
                                             len(filenames_neg['{}_lead{}'.format(ver, lead)])))
        
# ----------------------------------------------------- #
# Separate train and valid from pos / neg batches
filenames_pos_train = {}
filenames_neg_train = {}

filenames_pos_valid = {}
filenames_neg_valid = {}

for ver in vers:
    for lead in leads:
        temp_namelist_pos = filenames_pos['{}_lead{}'.format(ver, lead)]
        temp_namelist_neg = filenames_neg['{}_lead{}'.format(ver, lead)]
        
        pos_train, pos_valid = mu.name_extract(temp_namelist_pos)
        neg_train, neg_valid = mu.name_extract(temp_namelist_neg)
        
        print('pos train: {} pos valid: {} neg train: {} neg valid {}'.format(len(pos_train), len(pos_valid), len(neg_train),len(neg_valid)))
        
        filenames_pos_train['{}_lead{}'.format(ver, lead)] = pos_train
        filenames_neg_train['{}_lead{}'.format(ver, lead)] = neg_train
        
        filenames_pos_valid['{}_lead{}'.format(ver, lead)] = pos_valid
        filenames_neg_valid['{}_lead{}'.format(ver, lead)] = neg_valid

# ------------------------------------------------------------------ #
# Merge train/valid and pos/neg batch files from multiple lead times
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
        
# ----------------------------------------------------------------- #
# Load valid files for model training

filename_valid = neg_valid_all[::130] + pos_valid_all[::13]
L_valid = len(filename_valid)
print('number of validation batches: {}'.format(L_valid))

VALID_input_64 = np.empty((L_valid, 64, 64, L_vars))
VALID_target = np.ones(L_valid)

for i, name in enumerate(filename_valid):
    data = np.load(name)
    for k, c in enumerate(ind_pick_from_batch):
        
        VALID_input_64[i, ..., k] = data[..., c]

        if 'pos' in name:
            VALID_target[i] = 1.0
        else:
            VALID_target[i] = 0.0
            
# Save and load validation set to speed-up retraining
tuple_save = (VALID_input_64, VALID_target)
label_save = ['VALID_input_64', 'VALID_target']
du.save_hdf5(tuple_save, label_save, save_dir, 'CNN_Validation_vgg.hdf')

# with h5py.File(save_dir+'CNN_Validation_vgg.hdf', 'r') as h5io:
#     VALID_input_64 = h5io['VALID_input_64'][...]
#     VALID_target = h5io['VALID_target'][...]

model_head = mu.create_model_head(input_shape=(128,), N_node=64)
model_base = mu.create_model_vgg(input_shape=(32, 32, 15), channels = [32, 64, 96, 128])

IN = layers.Input(shape=(32, 32, 15))

VEC = model_base(IN)
OUT = model_head(VEC)

model_final = Model(inputs=IN, outputs=OUT)

# ============================================= #
# Weights

model_final.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer=keras.optimizers.Adam(lr=1e-4))
if model_prefix_load:
    W_old = mu.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}/'.format(model_prefix_load))    
    model_final.set_weights(W_old)


# ----------------------------------------------------------------- #
# model training loop
Y_pred = model_final.predict([VALID_input_64])
record_temp = mu.verif_metric(VALID_target, Y_pred)

# training parameters
epochs = 500
L_train = 64
min_del = 0.0
max_tol = 100 # early stopping with patience
batch_size = 200

# Allocate batch files
X_batch_64 = np.empty((batch_size, 64, 64, L_vars))
Y_batch = np.empty((batch_size, 1))
X_batch_64[...] = np.nan
Y_batch[...] = np.nan

# Model check-point info
model_name = model_prefix_save
model_path = temp_dir + model_name

# ========== Training loop ========== #
tol = 0 # initial tol

filename_pos_train = pos_train_all
filename_neg_train = neg_train_all
L_pos = len(filename_pos_train)
L_neg = len(filename_neg_train)

record = record_temp
print("Initial record: {}".format(record))

mu.set_seeds(seeds)
    
for i in range(epochs):
    start_time = time.time()

    # loop of batch
    for j in range(L_train):
        N_pos = 20
        N_neg = batch_size - N_pos

        ind_neg = du.shuffle_ind(L_neg)
        ind_pos = du.shuffle_ind(L_pos)
        
        # neg batches from this training rotation 
        file_pick_neg = []
        for ind_temp in ind_neg[:N_neg]:
            file_pick_neg.append(filename_neg_train[ind_temp])
        # pos batches from this training rotation 
        file_pick_pos = []
        for ind_temp in ind_pos[:N_pos]:
            file_pick_pos.append(filename_pos_train[ind_temp])
            
        # get all the batch filenames for checking labels
        file_pick = file_pick_neg + file_pick_pos

#         if len(file_pick) != batch_size:
#             sregwet # number of available files = batch size

        # Assign labels based on batch filenames
        for k in range(batch_size):
            data = np.load(file_pick[k])
            for l, c in enumerate(N_vars):
                temp = data[..., c] 
                X_batch_64[k, ..., l] = temp

            if 'pos' in file_pick[k]:
                Y_batch[k, :] = 1.0 #np.random.uniform(0.9, 0.99)
            elif 'neg_neg_neg' in file_pick[k]:
                Y_batch[k, :] = 0.0 #np.random.uniform(0.01, 0.05)
            else:
                werhgaer
        # ------------------------------------------------- #
        # batch input and label from this training rotation 
        ind_ = du.shuffle_ind(batch_size)
        X_batch_64 = X_batch_64[ind_, ...]
        Y_batch = Y_batch[ind_, :]

        # train on batch
        model_final.train_on_batch(X_batch_64, Y_batch);

    # epoch end operations
    Y_pred = model_final.predict([VALID_input_64])
    record_temp = mu.verif_metric(VALID_target, Y_pred)

    if (record - record_temp > min_del):
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        tol = 0
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

