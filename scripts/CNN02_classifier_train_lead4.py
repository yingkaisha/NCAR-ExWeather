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
parser.add_argument('lead1', help='lead1')
parser.add_argument('lead2', help='lead2')
parser.add_argument('lead3', help='lead3')
parser.add_argument('lead4', help='lead4')

parser.add_argument('lead_name', help='lead_name')
parser.add_argument('model_tag', help='model_tag')

args = vars(parser.parse_args())

# =============== #

lead1 = int(args['lead1'])
lead2 = int(args['lead2'])
lead3 = int(args['lead3'])
lead4 = int(args['lead4'])

lead_name = args['lead_name']
model_tag = args['model_tag']

L_vec = 8

# ================================================== #
# Functions

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
def verif_metric(VALID_target, Y_pred, ref):
    BS = np.mean((VALID_target.ravel() - Y_pred.ravel())**2)
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

def name_to_ind(filenames):
    
    indx_out = []
    indy_out = []
    day_out = []
    flag_out = []
    
    for i, name in enumerate(filenames):
        nums = re.findall(r'\d+', name)
        indy = int(nums[-2])
        indx = int(nums[-3])
        day = int(nums[-4])
        
        indx_out.append(indx)
        indy_out.append(indy)
        day_out.append(day)
        
        if "pos" in name:
            flag_out.append(True)
        else:
            flag_out.append(False)
        
    return np.array(indx_out), np.array(indy_out), np.array(day_out), np.array(flag_out)

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

def name_to_ind(filenames):
    
    indx_out = []
    indy_out = []
    day_out = []
    flag_out = []
    
    for i, name in enumerate(filenames):
        nums = re.findall(r'\d+', name)
        indy = int(nums[-2])
        indx = int(nums[-3])
        day = int(nums[-4])
        
        indx_out.append(indx)
        indy_out.append(indy)
        day_out.append(day)
        
        if "pos" in name:
            flag_out.append(True)
        else:
            flag_out.append(False)
        
    return np.array(indx_out), np.array(indy_out), np.array(day_out), np.array(flag_out)

def create_model():
    
    IN = keras.Input((L_vec, 128))
    X = IN
    X = keras.layers.Conv1D(128, kernel_size=2, strides=1, padding='valid')(X)
    X = keras.layers.Activation("gelu")(X)
    
    #
    IN_vec = keras.Input((2,))
    
    X = keras.layers.GlobalMaxPool1D()(X) #X = keras.layers.Flatten()(X)
    X = keras.layers.Concatenate()([X, IN_vec])
    
    X = keras.layers.Dense(64)(X)
    X = keras.layers.Activation("relu")(X)
    X = keras.layers.BatchNormalization()(X)

    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=[IN, IN_vec], outputs=OUT)
    return model

# ================================================================ #
# Geographical information

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

# ============================================================ #
# File path

filepath_vec = "/glade/work/ksha/NCAR/"

if (lead1 < 9) or (lead1 > 18):
    path_name1_v3 = '/glade/scratch/ksha/DATA/NCAR_batch_v3/'
else:
    path_name1_v3 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/'

if (lead2 < 9) or (lead2 > 18):
    path_name2_v3 = '/glade/scratch/ksha/DATA/NCAR_batch_v3/'
else:
    path_name2_v3 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/'
    
if (lead3 < 9) or (lead3 > 18):
    path_name3_v3 = '/glade/scratch/ksha/DATA/NCAR_batch_v3/'
else:
    path_name3_v3 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/'
    
if (lead4 < 9) or (lead4 > 18):
    path_name4_v3 = '/glade/scratch/ksha/DATA/NCAR_batch_v3/'
else:
    path_name4_v3 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/'

path_name1_v4 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'
path_name2_v4 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'
path_name3_v4 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'
path_name4_v4 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'

path_name1_v4_test = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'
path_name2_v4_test = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'
path_name3_v4_test = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'
path_name4_v4_test = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'

# ========================================================================= #
# Read batch file names (npy)

filename_train_lead1_v3 = sorted(glob("{}TRAIN*lead{}.npy".format(path_name1_v3, lead1)))
filename_train_lead2_v3 = sorted(glob("{}TRAIN*lead{}.npy".format(path_name2_v3, lead2)))
filename_train_lead3_v3 = sorted(glob("{}TRAIN*lead{}.npy".format(path_name3_v3, lead3)))
filename_train_lead4_v3 = sorted(glob("{}TRAIN*lead{}.npy".format(path_name4_v3, lead4)))

filename_valid_lead1_v3 = sorted(glob("{}VALID*lead{}.npy".format(path_name1_v3, lead1)))
filename_valid_lead2_v3 = sorted(glob("{}VALID*lead{}.npy".format(path_name2_v3, lead2)))
filename_valid_lead3_v3 = sorted(glob("{}VALID*lead{}.npy".format(path_name3_v3, lead3)))
filename_valid_lead4_v3 = sorted(glob("{}VALID*lead{}.npy".format(path_name4_v3, lead4)))

# ============================================================ #
# Consistency check indices

IND_TRAIN_lead = np.load('/glade/work/ksha/NCAR/IND_TRAIN_lead_full.npy', allow_pickle=True)[()]
TRAIN_ind1_v3 = IND_TRAIN_lead['lead{}'.format(lead1)]
TRAIN_ind2_v3 = IND_TRAIN_lead['lead{}'.format(lead2)]
TRAIN_ind3_v3 = IND_TRAIN_lead['lead{}'.format(lead3)]
TRAIN_ind4_v3 = IND_TRAIN_lead['lead{}'.format(lead4)]

IND_VALID_lead = np.load('/glade/work/ksha/NCAR/IND_VALID_lead_full.npy', allow_pickle=True)[()]
VALID_ind1_v3 = IND_VALID_lead['lead{}'.format(lead1)]
VALID_ind2_v3 = IND_VALID_lead['lead{}'.format(lead2)]
VALID_ind3_v3 = IND_VALID_lead['lead{}'.format(lead3)]
VALID_ind4_v3 = IND_VALID_lead['lead{}'.format(lead4)]

# ============================================================== #
# Load feature vectors (HRRR v3, training)

data_lead1_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead1_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead1_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]

data_lead2_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead2_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead2_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]

data_lead3_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead3_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead3_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]

data_lead4_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]
data_lead4_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]
data_lead4_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]

TRAIN_lead1_v3 = np.concatenate((data_lead1_p0['y_vector'], data_lead1_p1['y_vector'], data_lead1_p2['y_vector']), axis=0)
TRAIN_lead2_v3 = np.concatenate((data_lead2_p0['y_vector'], data_lead2_p1['y_vector'], data_lead2_p2['y_vector']), axis=0)
TRAIN_lead3_v3 = np.concatenate((data_lead3_p0['y_vector'], data_lead3_p1['y_vector'], data_lead3_p2['y_vector']), axis=0)
TRAIN_lead4_v3 = np.concatenate((data_lead4_p0['y_vector'], data_lead4_p1['y_vector'], data_lead4_p2['y_vector']), axis=0)

TRAIN_lead1_y_v3 = np.concatenate((data_lead1_p0['y_true'], data_lead1_p1['y_true'], data_lead1_p2['y_true']), axis=0)
TRAIN_lead2_y_v3 = np.concatenate((data_lead2_p0['y_true'], data_lead2_p1['y_true'], data_lead2_p2['y_true']), axis=0)
TRAIN_lead3_y_v3 = np.concatenate((data_lead3_p0['y_true'], data_lead3_p1['y_true'], data_lead3_p2['y_true']), axis=0)
TRAIN_lead4_y_v3 = np.concatenate((data_lead4_p0['y_true'], data_lead4_p1['y_true'], data_lead4_p2['y_true']), axis=0)

# =========================================================== #
# Load feature vectors (HRRR v3, validation)

data_lead1_valid = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead2_valid = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead3_valid = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead4_valid = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]

VALID_lead1_v3 = data_lead1_valid['y_vector']
VALID_lead2_v3 = data_lead2_valid['y_vector']
VALID_lead3_v3 = data_lead3_valid['y_vector']
VALID_lead4_v3 = data_lead4_valid['y_vector']

VALID_lead1_y_v3 = data_lead1_valid['y_true']
VALID_lead2_y_v3 = data_lead2_valid['y_true']
VALID_lead3_y_v3 = data_lead3_valid['y_true']
VALID_lead4_y_v3 = data_lead4_valid['y_true']

# ================================================================= #
# Collect feature vectors from all batch files (HRRR v3, validation)

L = len(TRAIN_ind2_v3)

filename_train1_pick_v3 = []
filename_train2_pick_v3 = []
filename_train3_pick_v3 = []
filename_train4_pick_v3 = []

TRAIN_X_lead1 = np.empty((L, 128))
TRAIN_X_lead2 = np.empty((L, 128))
TRAIN_X_lead3 = np.empty((L, 128))
TRAIN_X_lead4 = np.empty((L, 128))

TRAIN_Y_v3 = np.empty(L)

for i in range(L):
    
    ind_lead1_v3 = int(TRAIN_ind1_v3[i])
    ind_lead2_v3 = int(TRAIN_ind2_v3[i])
    ind_lead3_v3 = int(TRAIN_ind3_v3[i])
    ind_lead4_v3 = int(TRAIN_ind4_v3[i])
    
    filename_train1_pick_v3.append(filename_train_lead1_v3[ind_lead1_v3])
    filename_train2_pick_v3.append(filename_train_lead2_v3[ind_lead2_v3])
    filename_train3_pick_v3.append(filename_train_lead3_v3[ind_lead3_v3])
    filename_train4_pick_v3.append(filename_train_lead4_v3[ind_lead4_v3])
    
    TRAIN_X_lead1[i, :] = TRAIN_lead1_v3[ind_lead1_v3, :]
    TRAIN_X_lead2[i, :] = TRAIN_lead2_v3[ind_lead2_v3, :]
    TRAIN_X_lead3[i, :] = TRAIN_lead3_v3[ind_lead3_v3, :]
    TRAIN_X_lead4[i, :] = TRAIN_lead4_v3[ind_lead4_v3, :]
    
    TRAIN_Y_v3[i] = TRAIN_lead3_y_v3[ind_lead3_v3]
    
# ================================================================== #
# Collect feature vectors from all batch files (HRRR v3, validation)
L = len(VALID_ind2_v3)

filename_valid1_pick_v3 = []
filename_valid2_pick_v3 = []
filename_valid3_pick_v3 = []
filename_valid4_pick_v3 = []

VALID_X_lead1 = np.empty((L, 128))
VALID_X_lead2 = np.empty((L, 128))
VALID_X_lead3 = np.empty((L, 128))
VALID_X_lead4 = np.empty((L, 128))

VALID_Y_v3 = np.empty(L)

for i in range(L):
    
    ind_lead1_v3 = int(VALID_ind1_v3[i])
    ind_lead2_v3 = int(VALID_ind2_v3[i])
    ind_lead3_v3 = int(VALID_ind3_v3[i])
    ind_lead4_v3 = int(VALID_ind4_v3[i])
    
    filename_valid1_pick_v3.append(filename_valid_lead1_v3[ind_lead1_v3])
    filename_valid2_pick_v3.append(filename_valid_lead2_v3[ind_lead2_v3])
    filename_valid3_pick_v3.append(filename_valid_lead3_v3[ind_lead3_v3])
    filename_valid4_pick_v3.append(filename_valid_lead4_v3[ind_lead4_v3])
    
    VALID_X_lead1[i, :] = VALID_lead1_v3[ind_lead1_v3, :]
    VALID_X_lead2[i, :] = VALID_lead2_v3[ind_lead2_v3, :]
    VALID_X_lead3[i, :] = VALID_lead3_v3[ind_lead3_v3, :]
    VALID_X_lead4[i, :] = VALID_lead4_v3[ind_lead4_v3, :]
    
    VALID_Y_v3[i] = VALID_lead3_y_v3[ind_lead3_v3]

# ================================================================== #
# extract location information for nearby-grid-cell-based training

indx_train, indy_train, days_train, flags_train = name_to_ind(filename_train3_pick_v3)
indx_valid, indy_valid, days_valid, flags_valid = name_to_ind(filename_valid3_pick_v3)
grid_shape = lon_80km.shape

# ============================================= #
# Merge feature vectors on multiple lead times

N_days_train = np.max(days_train) + 1
N_days_valid = (np.max(days_valid) - np.min(days_valid) + 1) + 1

ALL_VEC = np.empty((N_days_train+N_days_valid, 4,)+grid_shape+(128,))
ALL_VEC[...] = np.nan

for i in range(len(indx_train)):
    indx_temp = indx_train[i]
    indy_temp = indy_train[i]
    days_temp = days_train[i]
    
    ALL_VEC[days_temp, 0, indx_temp, indy_temp, :] = TRAIN_X_lead1[i, :]
    ALL_VEC[days_temp, 1, indx_temp, indy_temp, :] = TRAIN_X_lead2[i, :]
    ALL_VEC[days_temp, 2, indx_temp, indy_temp, :] = TRAIN_X_lead3[i, :]
    ALL_VEC[days_temp, 3, indx_temp, indy_temp, :] = TRAIN_X_lead4[i, :]

for i in range(len(indx_valid)):
    indx_temp = indx_valid[i]
    indy_temp = indy_valid[i]
    days_temp = days_valid[i]
    
    ALL_VEC[days_temp, 0, indx_temp, indy_temp, :] = VALID_X_lead1[i, :]
    ALL_VEC[days_temp, 1, indx_temp, indy_temp, :] = VALID_X_lead2[i, :]
    ALL_VEC[days_temp, 2, indx_temp, indy_temp, :] = VALID_X_lead3[i, :]
    ALL_VEC[days_temp, 3, indx_temp, indy_temp, :] = VALID_X_lead4[i, :]

# ================================================#
# Combine training and validation set loation info

indx = np.concatenate((indx_train, indx_valid), axis=0)
indy = np.concatenate((indy_train, indy_valid), axis=0)
days = np.concatenate((days_train, days_valid), axis=0)
flags = np.concatenate((flags_train, flags_valid), axis=0)

# ======================================================== #
# Separate pos and neg samples for balanced training

TRAIN_Y = np.concatenate((TRAIN_Y_v3, VALID_Y_v3), axis=0)

TRAIN_pos_indx = indx[TRAIN_Y==1]
TRAIN_pos_indy = indy[TRAIN_Y==1]
TRAIN_pos_days = days[TRAIN_Y==1]

TRAIN_neg_indx = indx[TRAIN_Y==0]
TRAIN_neg_indy = indy[TRAIN_Y==0]
TRAIN_neg_days = days[TRAIN_Y==0]

lon_norm_v3, lat_norm_v3, elev_norm_v3, mon_norm_v3 = feature_extract(filename_train3_pick_v3, 
                                                                      lon_80km, lon_minmax, 
                                                                      lat_80km, lat_minmax, elev_80km, elev_max)

TRAIN_stn_v3 = np.concatenate((lon_norm_v3[:, None], 
                               lat_norm_v3[:, None]), axis=1)


lon_norm_v3, lat_norm_v3, elev_norm_v3, mon_norm_v3 = feature_extract(filename_valid3_pick_v3, 
                                                                      lon_80km, lon_minmax, 
                                                                      lat_80km, lat_minmax, elev_80km, elev_max)

VALID_stn_v3 = np.concatenate((lon_norm_v3[:, None], 
                               lat_norm_v3[:, None]), axis=1)

ALL_stn = np.concatenate((TRAIN_stn_v3, VALID_stn_v3))

TRAIN_stn_pos = ALL_stn[TRAIN_Y==1]
TRAIN_stn_neg = ALL_stn[TRAIN_Y==0]


# ====================================================== #
# HRRR v4x validation set
# ====================================================== #
# Read batch file names (npy)

filename_valid_lead1 = sorted(glob("{}TEST*lead{}.npy".format(path_name1_v4_test, lead1)))
filename_valid_lead2 = sorted(glob("{}TEST*lead{}.npy".format(path_name2_v4_test, lead2)))
filename_valid_lead3 = sorted(glob("{}TEST*lead{}.npy".format(path_name3_v4_test, lead3)))
filename_valid_lead4 = sorted(glob("{}TEST*lead{}.npy".format(path_name4_v4_test, lead4)))

# =============================== #
# Load feature vectors

valid_lead1 = np.load('{}TEST_v4_vec_lead{}_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
valid_lead2 = np.load('{}TEST_v4_vec_lead{}_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
valid_lead3 = np.load('{}TEST_v4_vec_lead{}_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
valid_lead4 = np.load('{}TEST_v4_vec_lead{}_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]

VALID_lead1 = valid_lead1['y_vector']
VALID_lead2 = valid_lead2['y_vector']
VALID_lead3 = valid_lead3['y_vector']
VALID_lead4 = valid_lead4['y_vector']

VALID_lead1_y = valid_lead1['y_true']
VALID_lead2_y = valid_lead2['y_true']
VALID_lead3_y = valid_lead3['y_true']
VALID_lead4_y = valid_lead4['y_true']

# ============================================================ #
# Consistency check indices

IND_TEST_lead = np.load('/glade/work/ksha/NCAR/IND_TEST_lead_v4.npy', allow_pickle=True)[()]

VALID_ind1 = IND_TEST_lead['lead{}'.format(lead1)]
VALID_ind2 = IND_TEST_lead['lead{}'.format(lead2)]
VALID_ind3 = IND_TEST_lead['lead{}'.format(lead3)]
VALID_ind4 = IND_TEST_lead['lead{}'.format(lead4)]

# ================================================================== #
# Collect feature vectors from all batch files

L = len(VALID_ind2)

filename_valid1_pick = []
filename_valid2_pick = []
filename_valid3_pick = []
filename_valid4_pick = []

VALID_X_lead1 = np.empty((L, 128))
VALID_X_lead2 = np.empty((L, 128))
VALID_X_lead3 = np.empty((L, 128))
VALID_X_lead4 = np.empty((L, 128))

VALID_Y = np.empty(L)

for i in range(L):
    
    ind_lead1 = int(VALID_ind1[i])
    ind_lead2 = int(VALID_ind2[i])
    ind_lead3 = int(VALID_ind3[i])
    ind_lead4 = int(VALID_ind4[i])
    
    filename_valid1_pick.append(filename_valid_lead1[ind_lead1])
    filename_valid2_pick.append(filename_valid_lead2[ind_lead2])
    filename_valid3_pick.append(filename_valid_lead3[ind_lead3])
    filename_valid4_pick.append(filename_valid_lead4[ind_lead4])
    
    VALID_X_lead1[i, :] = VALID_lead1[ind_lead1, :]
    VALID_X_lead2[i, :] = VALID_lead2[ind_lead2, :]
    VALID_X_lead3[i, :] = VALID_lead3[ind_lead3, :]
    VALID_X_lead4[i, :] = VALID_lead4[ind_lead4, :]
    
    VALID_Y[i] = VALID_lead3_y[ind_lead3]

# ================================================================== #
# extract location information
indx, indy, days, flags = name_to_ind(filename_valid3_pick)

lon_norm_v3, lat_norm_v3, elev_norm_v3, mon_norm_v3 = feature_extract(filename_valid3_pick, 
                                                                      lon_80km, lon_minmax, 
                                                                      lat_80km, lat_minmax, elev_80km, elev_max)

VALID_stn_v3 = np.concatenate((lon_norm_v3[:, None], 
                               lat_norm_v3[:, None]), axis=1)

# ================================================================== #
# Collect feature vectors from all batch files

grid_shape = lon_80km.shape
N_days = np.max(days)-np.min(days)+1

VALID_VEC = np.empty((N_days, 4,)+grid_shape+(128,))
VALID_VEC[...] = np.nan

for i in range(len(indx)):
    indx_temp = indx[i]
    indy_temp = indy[i]
    days_temp = days[i]-np.min(days)
    
    if days_temp <0:
        eqrgetwqh
    
    VALID_VEC[days_temp, 0, indx_temp, indy_temp, :] = VALID_X_lead1[i, :]
    VALID_VEC[days_temp, 1, indx_temp, indy_temp, :] = VALID_X_lead2[i, :]
    VALID_VEC[days_temp, 2, indx_temp, indy_temp, :] = VALID_X_lead3[i, :]
    VALID_VEC[days_temp, 3, indx_temp, indy_temp, :] = VALID_X_lead4[i, :]

# ========================================================================= #
# Generate v4 validation set with nearby grid cells
# ========================================================================= #

grid_shape = lon_80km.shape
day_min = np.min(days)
VALID_Input = np.empty((len(VALID_Y), L_vec, 128))
VALID_Input[...] = np.nan

VALID_Input_stn = np.empty((len(VALID_Y), 2))

for i in range(len(VALID_Y)):
    
    vec_merge = ()
    
    indx_temp = indx[i]
    indy_temp = indy[i]
    days_temp = days[i] - day_min
    
    indx_left = np.max([indx_temp - 1, 0])
    indx_right = np.min([indx_temp + 1, grid_shape[0]])
    
    indy_bot = np.max([indy_temp - 1, 0])
    indy_top = np.min([indy_temp + 1, grid_shape[1]])
    
    count = 0
    for ix in [indx_temp, indx_left, indx_right]:
        for iy in [indy_temp, indy_bot, indy_top]:
            for il in range(4):
                vec_temp = VALID_VEC[days_temp, il, ix, iy, :]
                if np.sum(np.isnan(vec_temp)) == 0 and count < L_vec:
                    vec_merge += (vec_temp[None, ...],)
                    count += 1
                    
    VEC_merge = np.concatenate(vec_merge, axis=0)
    VALID_Input[i, ...] = VEC_merge
    VALID_Input_stn[i, ...] = VALID_stn_v3[days_temp, :]
    
# ============================================================================== #
# Set randmo seeds

seeds = [12342, 2536234, 98765, 473, 865, 7456, 69472, 3456357, 3425, 678,
         2452624, 5787, 235362, 67896, 98454, 12445, 46767, 78906, 345, 8695, 
         2463725, 4734, 23234, 884, 2341, 362, 5, 234, 483, 785356, 23425, 3621, 
         58461, 80968765, 123, 425633, 5646, 67635, 76785, 34214, 9, 8, 7, 6, 5, 4, 3, 2, 1]

training_rounds = len(seeds)

# ============================================== #
# Model training
ref = np.sum(VALID_Y) / len(VALID_Y)
grid_shape = lon_80km.shape
day_min = np.min(TRAIN_neg_days)

# =========== Model Section ========== #

batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch/'
temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

key = '{}_lead{}'.format(model_tag, lead_name)

model_name = '{}'.format(key)
model_path = temp_dir+model_name

tol = 0

# ========== Training loop ========== #
L_pos = len(TRAIN_pos_indx)
L_neg = len(TRAIN_neg_indx)

record = 1.1
print("Initial record: {}".format(record))

min_del = 0
max_tol = 100 # early stopping with patience

epochs = 500
batch_size = 64
L_train = 16

N_pos = 32
N_neg = batch_size - N_pos

buffer = 10
N_pos += buffer
N_neg += buffer

N_neg_true = N_neg - buffer
N_pos_true = N_pos - buffer

NEG_Input = np.empty((N_neg, L_vec, 128))
NEG_Input[...] = np.nan

POS_Input = np.empty((N_pos, L_vec, 128))
POS_Input[...] = np.nan

NEG_Input_stn = np.empty((N_neg, 2))
POS_Input_stn = np.empty((N_neg, 2))

VEC_merge = np.empty((L_vec, 128))

for r in range(training_rounds):
    if r == 0:
        tol = 0
    else:
        tol = -200

    model = create_model()
    
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=keras.optimizers.Adam(lr=1e-4))
    
    set_seeds(int(seeds[r]))
    print('Training round {}'.format(r))

    
    for i in range(epochs):            
        start_time = time.time()

        # loop of batch
        for j in range(L_train):
            
            ind_neg = du.shuffle_ind(L_neg)
            ind_pos = du.shuffle_ind(L_pos)

            ind_neg_pick = ind_neg[:N_neg]
            ind_pos_pick = ind_pos[:N_pos]
            
            # ========== #
            count_save = 0
            for ineg, neg_pick in enumerate(ind_neg_pick):
                
                count = 0
                
                vec_merge = ()

                indx_temp = TRAIN_neg_indx[neg_pick]
                indy_temp = TRAIN_neg_indy[neg_pick]
                days_temp = TRAIN_neg_days[neg_pick] - day_min

                indx_left = np.max([indx_temp - 1, 0])
                indx_right = np.min([indx_temp + 1, grid_shape[0]])

                indy_bot = np.max([indy_temp - 1, 0])
                indy_top = np.min([indy_temp + 1, grid_shape[1]])
                
                for ix in [indx_temp, indx_left, indx_right]:
                    for iy in [indy_temp, indy_bot, indy_top]:
                        for il in range(4):
                            vec_temp = ALL_VEC[days_temp, il, ix, iy, :]
                            if np.sum(np.isnan(vec_temp)) == 0 and count < L_vec:
                                vec_merge += (vec_temp[None, ...],)
                                count += 1
                if count < L_vec:
                    continue
                else:
                    VEC_merge = np.concatenate(vec_merge, axis=0)
                    NEG_Input[count_save, ...] = VEC_merge
                    NEG_Input_stn[count_save, ...] = TRAIN_stn_neg[neg_pick]
                    count_save += 1
                    
            count_save = 0
            for ipos, pos_pick in enumerate(ind_pos_pick):
                count = 0
                
                vec_merge = ()

                indx_temp = TRAIN_pos_indx[pos_pick]
                indy_temp = TRAIN_pos_indy[pos_pick]
                days_temp = TRAIN_pos_days[pos_pick] - day_min

                indx_left = np.max([indx_temp - 1, 0])
                indx_right = np.min([indx_temp + 1, grid_shape[0]])

                indy_bot = np.max([indy_temp - 1, 0])
                indy_top = np.min([indy_temp + 1, grid_shape[1]])
                
                for ix in [indx_temp, indx_left, indx_right]:
                    for iy in [indy_temp, indy_bot, indy_top]:
                        for il in range(4):
                            vec_temp = ALL_VEC[days_temp, il, ix, iy, :]
                            if np.sum(np.isnan(vec_temp)) == 0 and count < L_vec:
                                vec_merge += (vec_temp[None, ...],)
                                count += 1
                if count < L_vec:
                    continue
                else:
                    VEC_merge = np.concatenate(vec_merge, axis=0)
                    POS_Input[count_save, ...] = VEC_merge
                    POS_Input_stn[count_save, ...] = TRAIN_stn_pos[pos_pick]
                    count_save += 1
            # ========== #
            
            X_batch = np.concatenate((NEG_Input[:N_neg_true, ...], POS_Input[:N_pos_true, ...]), axis=0)
            X_batch_stn = np.concatenate((NEG_Input_stn[:N_neg_true, ...], POS_Input_stn[:N_pos_true, ...]), axis=0)
            
            if np.sum(np.isnan(X_batch)) > 0:
                continue;
            
            Y_batch = np.ones([batch_size,])
            Y_batch[:N_neg] = 0.0
            
            # Y_batch = np.zeros([batch_size,])
            # Y_batch[:N_neg] = np.random.uniform(low=0.0, high=0.1, size=N_neg)
            # Y_batch[N_neg:] = np.random.uniform(low=0.95, high=1.0, size=N_pos)

            ind_ = du.shuffle_ind(batch_size)

            X_batch = X_batch[ind_, :]
            Y_batch = Y_batch[ind_]
            
            model.train_on_batch([X_batch, X_batch_stn], Y_batch);

        # epoch end operations
        Y_pred = model.predict([VALID_Input, VALID_Input_stn])

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
            if record_temp > 1.1:
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
        
# ================================================================================ #
# Inference
    
model = create_model()

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=keras.optimizers.Adam(lr=0))

W_old = k_utils.dummy_loader('/glade/work/ksha/NCAR/Keras_models/{}_lead{}'.format(model_tag, lead_name))
model.set_weights(W_old)
        
Y_pred_valid = model.predict([VALID_Input, VALID_Input_stn])

# Save results
save_dict = {}
save_dict['Y_pred_valid'] = Y_pred_valid
save_dict['VALID_Y'] = VALID_Y

np.save('{}RESULT_lead{}_{}.npy'.format(filepath_vec, lead_name, model_tag), save_dict)
print('{}RESULT_lead{}_{}.npy'.format(filepath_vec, lead_name, model_tag))