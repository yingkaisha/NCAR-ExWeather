
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
print('Generating NEG batches from lead{}'.format(lead))

HRRRv3_lead = zarr.load(save_dir_scratch+'HRRR_{:02}_v3.zarr'.format(lead))

with h5py.File(save_dir_scratch+'SPC_to_lead{}.hdf'.format(lead), 'r') as h5io:
    record_v3 = h5io['record_v3'][...]
    
with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_72km = h5io['lon_72km'][...]
    lat_72km = h5io['lat_72km'][...]

with h5py.File(save_dir+'HRRR_torn_monthly.hdf', 'r') as h5io:
    clim = h5io['clim'][:]
    
means = [
    -6.335041783675384,
    101598.30648208999,
    2.4340308170812857,
    0.0238316214287872,
    0.0115228964831135,
    0.015723252607236175,
    0.00010298927478466365,
    0.00013315081911787703,
    0.02022990418418194,
    285.1588453352469,
    280.69456763975046,
    0.18025322895802864,
    -0.35625256772098957,
    4.466962100212334,
    0.10710428466431396,
    311.51020050786116,
    -22.95554152474839,
    95.80303950026172,
    41.22773039479408,
    2.696538199313979,
    0.257023643073863,
    11.80181492281666,
    0.15778718430103703,
];

stds = [
    8.872575669978966,
    672.3339463894478,
    7.555104640235371,
    0.5696550725786566,
    0.2283199203388272,
    0.37333362094670486,
    0.00022281640603195643,
    0.0002413561909874066,
    0.3589573748563584,
    11.553795616392204,
    12.101590155483459,
    3.1758721705443826,
    3.6588052023281175,
    2.6995797278745948,
    0.9896017905552607,
    748.8376068157106,
    78.895180023938,
    104.17948262883918*2,
    77.25788246299936*2,
    5.35086729614372,
    5.438075471238217,
    11.440203318938076,
    11.327741531273508
];

ind_pick = [0, 1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

log_norm = [True, False, True, True, True, True, True, True, True, False, False, 
            False, False, True, True, True, True, False, False, False, False, False, False]

sparse = [True, False, True, True, True, True, True, True, True, False, False, 
          False, False, False, True, True, True, False, False, False, False, False, False]

grid_shape = record_v3.shape

L_vars = len(ind_pick) + 1

target_size = 24
input_size = 128

half_margin = int((input_size - target_size) / 2)

grid_shape_input = (1059, 1799)

batch_dir_neg = '/glade/scratch/ksha/DATA/NCAR_batch_neg/'
prefix_train = 'TRAIN_neg_{}_mag{}_lead'+str(lead)+'.npy'
prefix_valid = 'VALID_neg_{}_mag{}_lead'+str(lead)+'.npy'

L_train = np.min([675, grid_shape[0]])
L_valid = grid_shape[0]-L_train

count = 0
out_slice = np.empty((1, input_size, input_size, L_vars))

base_v3_s = datetime(2018, 7, 15)
base_v3_e = datetime(2020, 12, 2)

base_v4_s = datetime(2020, 12, 3)
base_v4_e = datetime(2022, 7, 15)

base_ref = datetime(2010, 1, 1)

date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+180)]

for i in range(grid_shape[0]-1):

    if i < L_train:
        prefix = prefix_train
    else:
        prefix = prefix_valid

    lon_temp = record_v3[i, 0]
    lat_temp = record_v3[i, 1]
    mag_temp = record_v3[i, 2]

    flag_obs = lon_temp + lat_temp
    
    # if day i is positive 
    if np.logical_not(np.isnan(flag_obs)):
        
        # try day (i+1)
        k = i + 1
        date_time = date_list_v3[k]
        mon_ind = date_time.month - 1
        
        lon_temp_neg = record_v3[k, 0]
        lat_temp_neg = record_v3[k, 1]
        
        # if day 1+1 is negative, flag should be NaN
        flag_obs_neg = lon_temp_neg + lat_temp_neg
        
        indx_3km, indy_3km = du.grid_search(lon_3km, lat_3km, np.array(lon_temp)[None], np.array(lat_temp)[None])
        indx_3km = indx_3km[0]
        indy_3km = indy_3km[0]
        
        while np.isnan(flag_obs_neg) and k < grid_shape[0]:
            
            for augx in range(0, target_size, 4):
                for augy in range(0, target_size, 4):

                    x_margin_left = augx
                    x_margin_right = target_size - augx

                    y_margin_bottom = augy
                    y_margin_top = target_size - augy

                    x_edge_left = indx_3km - x_margin_left - half_margin
                    x_edge_right = indx_3km + x_margin_right + half_margin

                    y_edge_bottom = indy_3km - y_margin_bottom - half_margin
                    y_edge_top = indy_3km + y_margin_top + half_margin

                    if x_edge_left >= 0 and y_edge_bottom >= 0 and x_edge_right <= grid_shape_input[0] and y_edge_top <= grid_shape_input[1]:

                        for v, ind_var in enumerate(ind_pick):
                            
                            temp = HRRRv3_lead[k, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, ind_var]
                            
                            if ind_var == 0:
                                temp[temp<0] = 0
                            
                            if log_norm[ind_var]:
                                temp = np.log(np.abs(temp)+1)
                            else:
                                temp = (temp - means[ind_var])/stds[ind_var]
                            
                            out_slice[..., v] = temp
                        
                        out_slice[..., L_vars-1] = clim[mon_ind, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top]
                        
                        if np.sum(np.isnan(out_slice)) > 0:
                            print('HRRR contains NaN')
                            continue;
                        else:
                            save_name = batch_dir_neg+prefix.format(count, int(mag_temp))
                            print(save_name)
                            np.save(save_name, out_slice)
                            count += 1

            k += 1
            if k < grid_shape[0]:
                lon_temp_neg = record_v3[k, 0]
                lat_temp_neg = record_v3[k, 1]

                flag_obs_neg = lon_temp_neg + lat_temp_neg
            else:
                flag_obs_neg = 999





