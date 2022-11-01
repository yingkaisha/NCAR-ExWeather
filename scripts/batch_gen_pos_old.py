
import sys
from glob import glob

import time
import h5py
import zarr
import numpy as np
import pandas as pd

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

from datetime import datetime, timedelta
import subprocess

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead', help='lead')
args = vars(parser.parse_args())

# =============== #
lead = int(args['lead'])
print('Generating batches from lead{}'.format(lead))

HRRRv3_lead = zarr.load(save_dir_scratch+'HRRR_{:02}_v3.zarr'.format(lead))

with h5py.File(save_dir_scratch+'SPC_to_lead{}.hdf'.format(lead), 'r') as h5io:
    record_v3 = h5io['record_v3'][...]
    
with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_72km = h5io['lon_72km'][...]
    lat_72km = h5io['lat_72km'][...]
    
features_pick = [10, 13, 18, 19, 20]

mean_dp = 280
std_dp = 12

mean_srh = 41
std_srh = 80

mean_ushear = 3
std_ushear = 5

mean_vshear = 0.2
std_vshear = 5

grid_shape = record_v3.shape

L_vars = len(features_pick)

target_size = 24
input_size = 128

half_margin = int((input_size - target_size) / 2)

grid_shape_input = (1059, 1799)

batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch/'
prefix_train = 'TRAIN_pos_{}_mag{}_lead'+str(lead)+'.npy'
prefix_valid = 'VALID_pos_{}_mag{}_lead'+str(lead)+'.npy'

L_train = np.min([675, grid_shape[0]])
L_valid = grid_shape[0]-L_train

count = 0
out_slice = np.empty((1, input_size, input_size, L_vars))

for f in range(int(grid_shape[-1]/3)):
    for i in range(grid_shape[0]):
        
        if i < L_train:
            prefix = prefix_train
        else:
            prefix = prefix_valid
        
        lon_temp = record_v3[i, 3*f]
        lat_temp = record_v3[i, 3*f+1]
        mag_temp = record_v3[i, 3*f+2]
        
        flag_obs = lon_temp + lat_temp

        if np.logical_not(np.isnan(flag_obs)):

            indx_3km, indy_3km = du.grid_search(lon_3km, lat_3km, np.array(lon_temp)[None], np.array(lat_temp)[None])
            indx_3km = indx_3km[0]
            indy_3km = indy_3km[0]

            for augx in range(target_size):
                for augy in range(target_size):

                    x_margin_left = augx
                    x_margin_right = target_size - augx

                    y_margin_bottom = augy
                    y_margin_top = target_size - augy

                    x_edge_left = indx_3km - x_margin_left - half_margin
                    x_edge_right = indx_3km + x_margin_right + half_margin

                    y_edge_bottom = indy_3km - y_margin_bottom - half_margin
                    y_edge_top = indy_3km + y_margin_top + half_margin

                    if x_edge_left >= 0 and y_edge_bottom >= 0 and x_edge_right <= grid_shape_input[0] and y_edge_top <= grid_shape_input[1]:

                        for v, ind_var in enumerate(features_pick):
                            out_slice[..., v] = HRRRv3_lead[i, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, ind_var]
                        
                        if np.sum(np.isnan(out_slice))>0:
                            print('HRRR contains NaN')
                            continue;
                        else:  
                        
                            # ----- Normalization ----- #
                            out_slice[..., 0] = (out_slice[..., 0] - mean_dp) / std_dp
                            out_slice[..., 1] = np.log(out_slice[..., 1]+1)
                            out_slice[..., 2] = (out_slice[..., 2] - mean_srh) / std_srh
                            out_slice[..., 3] = (out_slice[..., 3] - mean_ushear) / std_ushear
                            out_slice[..., 4] = (out_slice[..., 4] - mean_vshear) / std_vshear
                            # ------------------------- #

                            save_name = batch_dir+prefix.format(count, int(mag_temp))
                            print(save_name)
                            np.save(save_name, out_slice)

                            count += 1

                    else:
                        continue;


