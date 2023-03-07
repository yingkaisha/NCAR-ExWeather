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

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('lead', help='lead')

args = vars(parser.parse_args())

# ===================================== #
lead = int(args['lead'])

filepath_vec = "/glade/work/ksha/NCAR/"

path_name1_v3 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/'
path_name2_v3 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/' 
path_name3_v3 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/'
path_name4_v3 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/'

path_name1_v4 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'
path_name2_v4 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'
path_name3_v4 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'
path_name4_v4 = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'

path_name1_v4_test = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'
path_name2_v4_test = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'
path_name3_v4_test = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'
path_name4_v4_test = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'

filename_train_lead_v3 = sorted(glob("{}TRAIN*lead{}.npy".format(path_name1_v3, lead)))
filename_valid_lead_v3 = sorted(glob("{}VALID*lead{}.npy".format(path_name1_v3, lead)))
filename_test_lead = sorted(glob("{}TEST*lead{}.npy".format(path_name1_v4_test, lead)))

L_train = len(filename_train_lead_v3)
L_valid = len(filename_valid_lead_v3)
L_test = len(filename_test_lead)

L = L_train+L_valid
filename_train = filename_train_lead_v3+filename_valid_lead_v3
filename_test = filename_test_lead

TRAIN_MAX = np.empty((L, 15))
TRAIN_MEAN = np.empty((L, 15))

TEST_MAX = np.empty((L_test, 15))
TEST_MEAN = np.empty((L_test, 15))

for i in range(L):
    temp_data = np.load(filename_train[i])
    
    for c in range(15):
        TRAIN_MAX[i, c] = np.max(temp_data[..., c])
        TRAIN_MEAN[i, c] = np.mean(temp_data[..., c])
        
for i in range(L_test):
    temp_data = np.load(filename_test[i])
    
    for c in range(15):
        TEST_MAX[i, c] = np.max(temp_data[..., c])
        TEST_MEAN[i, c] = np.mean(temp_data[..., c])

save_dict = {}
save_dict['TRAIN_MAX'] = TRAIN_MAX
save_dict['TRAIN_MEAN'] = TRAIN_MEAN

save_dict['TEST_MAX'] = TEST_MAX
save_dict['TEST_MEAN'] = TEST_MEAN

np.save('/glade/work/ksha/NCAR/TRAIN_MINMAX_lead{}.npy'.format(lead), save_dict)
print('/glade/work/ksha/NCAR/TRAIN_MINMAX_lead{}.npy'.format(lead))
