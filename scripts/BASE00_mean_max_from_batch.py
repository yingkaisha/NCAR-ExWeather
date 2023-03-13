'''
Compute the max and mean values from each 64-by-64 training batch and all its predictors.
The max and mean values are applied to the development of an MLP baseline
'''

# general tools
import os
import re
import sys
import time
import numpy as np
from glob import glob

from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
args = vars(parser.parse_args())

# =============================================== #
# obtain the current forecast lead time
lead = int(args['lead'])
N_vars = 15 # Number of predictors in batch file
# =============================================== #
# File path
filepath_vec = "/glade/work/ksha/NCAR/"
# HRRR v3
path_name1_v3 = path_batch_v3
path_name2_v3 = path_batch_v3
path_name3_v3 = path_batch_v3
path_name4_v3 = path_batch_v3

# Pull file sequence from HRRR v3 path
filename_train = sorted(glob("{}TRAIN*lead{}.npy".format(path_name1_v3, lead)))
filename_valid = sorted(glob("{}VALID*lead{}.npy".format(path_name1_v3, lead)))
filename_test = sorted(glob("{}TEST*lead{}.npy".format(path_name1_v4_test, lead)))

# Estimate the total number of batch files
L_train = len(filename_train)
L_valid = len(filename_valid)
L_test = len(filename_test)

# Array allocation 
TRAIN_MAX = np.empty((L_train, 15)); TRAIN_MAX[...] = np.nan
TRAIN_MEAN = np.empty((L_train, 15)); TRAIN_MEAN[...] = np.nan

VALID_MAX = np.empty((L_valid, 15)); VALID_MAX[...] = np.nan
VALID_MEAN = np.empty((L_valid, 15)); VALID_MEAN[...] = np.nan

TEST_MAX = np.empty((L_test, 15)); TEST_MAX[...] = np.nan
TEST_MEAN = np.empty((L_test, 15)); TEST_MEAN[...] = np.nan

# Load and store mean-max values
# training batches
for i in range(L_train):
    temp_data = np.load(filename_train[i])
    for c in range(N_vars):
        TRAIN_MAX[i, c] = np.max(temp_data[..., c])
        TRAIN_MEAN[i, c] = np.mean(temp_data[..., c])
# valid batches
for i in range(L_valid):
    temp_data = np.load(filename_valid[i])
    for c in range(N_vars):
        VALID_MAX[i, c] = np.max(temp_data[..., c])
        VALID_MEAN[i, c] = np.mean(temp_data[..., c])
# Testing batches
for i in range(L_test):
    temp_data = np.load(filename_test[i])
    for c in range(15):
        TEST_MAX[i, c] = np.max(temp_data[..., c])
        TEST_MEAN[i, c] = np.mean(temp_data[..., c])

# Save npy pickle
save_dict = {}
save_dict['TRAIN_MAX'] = TRAIN_MAX
save_dict['TRAIN_MEAN'] = TRAIN_MEAN

save_dict['VALID_MAX'] = VALID_MAX
save_dict['VALID_MEAN'] = VALID_MEAN

save_dict['TEST_MAX'] = TEST_MAX
save_dict['TEST_MEAN'] = TEST_MEAN


np.save('/glade/work/ksha/NCAR/TRAIN_MINMAX_lead{}.npy'.format(lead), save_dict)
print('/glade/work/ksha/NCAR/TRAIN_MINMAX_lead{}.npy'.format(lead))
