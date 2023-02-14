
# general tools
import os
import re
import sys
import time
import h5py
import random
from glob import glob

import numpy as np
from datetime import datetime, timedelta
from random import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from keras_unet_collection import utils as k_utils

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
args = vars(parser.parse_args())

# =============== #
lead = int(args['lead'])
print('Predict lead {}'.format(lead))


def verif_metric(VALID_target, Y_pred, ref):


    # fpr, tpr, thresholds = roc_curve(VALID_target.ravel(), Y_pred.ravel())
    # AUC = auc(fpr, tpr)
    # AUC_metric = 1 - AUC
    
    BS = np.mean((VALID_target.ravel() - Y_pred.ravel())**2)
    #ll = log_loss(VALID_target.ravel(), Y_pred.ravel())
    
    #print('{}'.format(BS))
    metric = BS

    return metric / ref

def feature_extract(filenames, lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max):
    
    lon_out = []
    lat_out = []
    elev_out = []
    mon_out = []
    
    base_v3_s = datetime(2018, 7, 15)
    base_v3_e = datetime(2020, 12, 2)

    base_v4_s = datetime(2020, 12, 3)
    base_v4_e = datetime(2022, 7, 15)

    base_ref = datetime(2010, 1, 1)
    
    date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
    date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+180-151)]
    
    for i, name in enumerate(filenames):
        
        if 'v4' in name:
            date_list = date_list_v4
        else:
            date_list = date_list_v3
        
        nums = re.findall(r'\d+', name)
        indy = int(nums[-2])
        indx = int(nums[-3])
        day = int(nums[-4])
        day = date_list[day]
        month = day.month
        
        month_norm = (month - 1)/(12-1)
        
        lon = lon_80km[indx, indy]
        lat = lat_80km[indx, indy]

        lon = (lon - lon_minmax[0])/(lon_minmax[1] - lon_minmax[0])
        lat = (lat - lat_minmax[0])/(lat_minmax[1] - lat_minmax[0])

        elev = elev_80km[indx, indy]
        elev = elev / elev_max
        
        lon_out.append(lon)
        lat_out.append(lat)
        elev_out.append(elev)
        mon_out.append(month_norm)
        
    return np.array(lon_out), np.array(lat_out), np.array(elev_out), np.array(mon_out)

def create_model():
    
    IN_vec = keras.Input((128,))
    
    IN_elev = keras.Input((3,))
    
    X_elev = IN_elev
    
    # X_elev = keras.layers.Dense(32, activity_regularizer=keras.regularizers.L2(1e-2))(X_elev)
    # X_elev = keras.layers.BatchNormalization()(X_elev)
    # X_elev = keras.layers.Activation("gelu")(X_elev)
    
    IN = keras.layers.Concatenate()([X_elev, IN_vec])
    
    X = IN
    #
    X = keras.layers.Dense(1024, activity_regularizer=keras.regularizers.L2(1e-2))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation("gelu")(X)
    #X = keras.layers.Activation("relu")(X)

    X = keras.layers.Dropout(0.3)(X)
    
    #
    X = keras.layers.Dense(512, activity_regularizer=keras.regularizers.L2(1e-2))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation("gelu")(X)
    #X = keras.layers.Activation("relu")(X)
    
    X = keras.layers.Dropout(0.3)(X)
    
    X = keras.layers.Dense(128, activity_regularizer=keras.regularizers.L2(1e-2))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation("gelu")(X)
    #X = keras.layers.Activation("relu")(X)
    
    #X = keras.layers.Dropout(0.3)(X)

    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=[IN_vec, IN_elev], outputs=OUT)
    
    return model

with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_80km = h5io['lon_80km'][...]
    lat_80km = h5io['lat_80km'][...]
    elev_3km = h5io['elev_3km'][...]
    land_mask_80km = h5io['land_mask_80km'][...]
    
grid_shape = land_mask_80km.shape

elev_80km = du.interp2d_wraper(lon_3km, lat_3km, elev_3km, lon_80km, lat_80km, method='linear')

elev_80km[np.isnan(elev_80km)] = 0
elev_80km[elev_80km<0] = 0
elev_max = np.max(elev_80km)

lon_80km_mask = lon_80km[land_mask_80km]
lat_80km_mask = lat_80km[land_mask_80km]

lon_minmax = [np.min(lon_80km_mask), np.max(lon_80km_mask)]
lat_minmax = [np.min(lat_80km_mask), np.max(lat_80km_mask)]

filepath_vec = "/glade/work/ksha/NCAR/"

filename_train_lead = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/TRAIN*lead{}.npy".format(lead)))

IND_TRAIN_lead = np.load('/glade/work/ksha/NCAR/IND_TRAIN_lead.npy', allow_pickle=True)[()]
TRAIN_ind = IND_TRAIN_lead['lead{}'.format(lead)]

data_lead_p0 = np.load('{}TRAIN_pp15_pred_lead{}_part0_base.npy'.format(filepath_vec, lead), allow_pickle=True)[()]
data_lead_p1 = np.load('{}TRAIN_pp15_pred_lead{}_part1_base.npy'.format(filepath_vec, lead), allow_pickle=True)[()]
data_lead_p2 = np.load('{}TRAIN_pp15_pred_lead{}_part2_base.npy'.format(filepath_vec, lead), allow_pickle=True)[()]

TRAIN_lead = np.concatenate((data_lead_p0['y_vector'], data_lead_p1['y_vector'], data_lead_p2['y_vector']), axis=0)
TRAIN_lead_y = np.concatenate((data_lead_p0['y_true'], data_lead_p1['y_true'], data_lead_p2['y_true']), axis=0)

L = len(TRAIN_ind)

filename_train_pick = []

TRAIN_X = np.empty((L, 128))
TRAIN_Y = np.empty(L)

for i in range(L):
    
    ind_lead = int(TRAIN_ind[i])
    filename_train_pick.append(filename_train_lead[ind_lead])
    
    TRAIN_X[i, ...] = TRAIN_lead[ind_lead, :]
    TRAIN_Y[i] = TRAIN_lead_y[ind_lead]
    
lon_norm, lat_norm, elev_norm, mon_norm = feature_extract(
    filename_train_pick, lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)

TRAIN_stn = np.concatenate((lon_norm[:, None], lat_norm[:, None], elev_norm[:, None]), axis=1)
TRAIN_merge = TRAIN_stn

TRAIN_256_pos = TRAIN_X[TRAIN_Y==1, :]
TRAIN_256_neg = TRAIN_X[TRAIN_Y==0, :]

TRAIN_stn_pos = TRAIN_merge[TRAIN_Y==1]
TRAIN_stn_neg = TRAIN_merge[TRAIN_Y==0]

filename_valid_lead = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v3/VALID*lead{}.npy".format(lead)))
valid_lead = np.load('{}TEST_pp15_pred_lead{}_base.npy'.format(filepath_vec, lead), allow_pickle=True)[()]
VALID_lead = valid_lead['y_vector']
VALID_lead_y = valid_lead['y_true']

data_p_valid = np.load('{}TEST_pp15_pred_lead{}_base.npy'.format(filepath_vec, lead), allow_pickle=True)[()]

VALID_256 = data_p_valid['y_vector']
VALID_pred = data_p_valid['y_pred']
VALID_Y = data_p_valid['y_true']

IND_VALID_lead = np.load('/glade/work/ksha/NCAR/IND_VALID_lead.npy', allow_pickle=True)[()]

VALID_ind = IND_VALID_lead['lead{}'.format(lead)]

L = len(VALID_ind)

filename_valid_pick = []

VALID_X = np.empty((L, 128))
VALID_Y = np.zeros(L)

for i in range(L):
    
    ind_lead = int(VALID_ind[i])
    
    filename_valid_pick.append(filename_valid_lead[ind_lead])
    
    VALID_X[i, ...]   = VALID_lead[ind_lead, :]

    if 'pos' in filename_valid_lead[ind_lead]:
        if VALID_lead_y[ind_lead] == 1.0:
            VALID_Y[i] = 1.0
        else:
            egwrshat

lon_norm, lat_norm, elev_norm, mon_norm = feature_extract(
    filename_valid_pick, lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)

VALID_stn = np.concatenate((lon_norm[:, None], lat_norm[:, None], elev_norm[:, None]), axis=1)
VALID_merge = VALID_stn

ref = np.sum(VALID_Y) / len(VALID_Y)

# =========== Model Section ========== #

batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch/'
temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

key = 'STN_Lead{}'.format(lead)

model_name = '{}'.format(key)
model_path = temp_dir+model_name

tol = 0

# ========== Training loop ========== #
L_pos = len(TRAIN_256_pos)
L_neg = len(TRAIN_256_neg)

record = 1.1
print("Initial record: {}".format(record))

min_del = 0
max_tol = 100 # early stopping with patience

epochs = 500
batch_size = 64
L_train = 16 #int(len(TRAIN_Y_pick) / batch_size)

tol = 0
model = create_model()
#
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=keras.optimizers.Adam(lr=1e-4))

# W_old = k_utils.dummy_loader('/glade/work/ksha/NCAR/Keras_models/BASE_Lead2/')
# model.set_weights(W_old)

for i in range(epochs):
    # if i > 0:
    #     backend.set_value(model.optimizer.learning_rate, decayed_learning_rate(i))

    start_time = time.time()

    # loop of batch
    for j in range(L_train):
        N_pos = 32
        N_neg = batch_size - N_pos

        ind_neg = du.shuffle_ind(L_neg)
        ind_pos = du.shuffle_ind(L_pos)

        ind_neg_pick = ind_neg[:N_neg]
        ind_pos_pick = ind_pos[:N_pos]

        X_batch_neg = TRAIN_256_neg[ind_neg_pick, :]
        X_batch_pos = TRAIN_256_pos[ind_pos_pick, :]

        X_batch_stn_neg = TRAIN_stn_neg[ind_neg_pick, :]
        X_batch_stn_pos = TRAIN_stn_pos[ind_pos_pick, :]

        X_batch = np.concatenate((X_batch_neg, X_batch_pos), axis=0)
        X_batch_stn = np.concatenate((X_batch_stn_neg, X_batch_stn_pos), axis=0)

        Y_batch = np.ones([batch_size,])
        Y_batch[:N_neg] = 0.0

        ind_ = du.shuffle_ind(batch_size)

        X_batch = X_batch[ind_, :]
        X_batch_stn = X_batch_stn[ind_, :]
        Y_batch = Y_batch[ind_]

        # train on batch
        model.train_on_batch([X_batch, X_batch_stn], Y_batch);

    # epoch end operations
    Y_pred = model.predict([VALID_X, VALID_merge])

    Y_pred[Y_pred<0] = 0
    Y_pred[Y_pred>1] = 1

    record_temp = verif_metric(VALID_Y, Y_pred, ref)

    # if i % 10 == 0:
    #     model.save(model_path_backup)

    if (record - record_temp > min_del):
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        tol = 0

        #print('tol: {}'.format(tol))
        # save
        print('save to: {}'.format(model_path))
        model.save(model_path)
    else:
        print('Validation loss {} NOT improved'.format(record_temp))
        if record_temp > 1.0:
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
            

model = create_model()

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=keras.optimizers.Adam(lr=0))

W_old = k_utils.dummy_loader('/glade/work/ksha/NCAR/Keras_models/STN_Lead{}/'.format(lead))
model.set_weights(W_old)

ref = np.sum(VALID_Y) / len(VALID_Y)
Y_pred = model.predict([VALID_X, VALID_merge])
record_temp = verif_metric(VALID_Y, Y_pred, ref)

save_dict = {}
save_dict['Y_pred'] = Y_pred
save_dict['VALID_Y'] = VALID_Y
np.save('{}RESULT_STN_lead{}_base.npy'.format(filepath_vec, lead), save_dict)
print('{}RESULT_STN_lead{}_base.npy'.format(filepath_vec, lead))
