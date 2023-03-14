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
import model_utils as mu

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
path_name1_v3 = path_batch_v3
path_name2_v3 = path_batch_v3
path_name3_v3 = path_batch_v3
path_name4_v3 = path_batch_v3

path_name1_v4 = path_batch_v4x
path_name2_v4 = path_batch_v4x
path_name3_v4 = path_batch_v4x
path_name4_v4 = path_batch_v4x

path_name1_v4_test = path_batch_v4
path_name2_v4_test = path_batch_v4
path_name3_v4_test = path_batch_v4
path_name4_v4_test = path_batch_v4

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

indx_train, indy_train, days_train, flags_train = mu.name_to_ind(filename_train3_pick_v3)
indx_valid, indy_valid, days_valid, flags_valid = mu.name_to_ind(filename_valid3_pick_v3)
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
    
# ======================================================== #
# Separate pos and neg samples for balanced training

TRAIN_Y = np.concatenate((TRAIN_Y_v3, VALID_Y_v3), axis=0)

TRAIN_pos_x = ALL_VEC[TRAIN_Y==1]
TRAIN_neg_x = ALL_VEC[TRAIN_Y==0]

lon_norm_v3, lat_norm_v3, elev_norm_v3, mon_norm_v3 = mu.feature_extract(filename_train3_pick_v3, 
                                                 lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)

TRAIN_stn_v3 = np.concatenate((lon_norm_v3[:, None], lat_norm_v3[:, None]), axis=1)

lon_norm_v3, lat_norm_v3, elev_norm_v3, mon_norm_v3 = mu.feature_extract(filename_valid3_pick_v3, 
                                                 lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)

VALID_stn_v3 = np.concatenate((lon_norm_v3[:, None], lat_norm_v3[:, None]), axis=1)

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
indx, indy, days, flags = mu.name_to_ind(filename_valid3_pick)

lon_norm_v3, lat_norm_v3, elev_norm_v3, mon_norm_v3 = mu.feature_extract(filename_valid3_pick, 
                                                 lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max)

VALID_stn_v3 = np.concatenate((lon_norm_v3[:, None], lat_norm_v3[:, None]), axis=1)

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

# TRAIN_pos_x & TRAIN_stn_neg & TRAIN_stn_pos & TRAIN_stn_neg
#VALID_VEC & VALID_Y & VALID_stn_v3





