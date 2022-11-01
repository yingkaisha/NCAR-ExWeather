
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

def neighbour_leads(lead):
    out = [lead-2, lead-1, lead, lead+1]
    flag_shift = [0, 0, 0, 0]
    
    for i in range(4):
        if out[i] < 0:
            out[i] = 24+out[i]
            flag_shift[i] = -1
        if out[i] > 23:
            out[i] = out[i]-24
            flag_shift[i] = +1
            
    return out, flag_shift

HRRRv3_lead = zarr.load(save_dir_scratch+'HRRR_{:02}_v3.zarr'.format(lead))
#HRRRv3_lead = np.zeros((872, 1059, 1799, 23)) # use fake data for a quick test
    
with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_72km = h5io['lon_72km'][...]
    lat_72km = h5io['lat_72km'][...]
    land_mask_72km = h5io['land_mask_72km'][...]
    land_mask_3km = h5io['land_mask_3km'][...]

# with h5py.File(save_dir+'HRRR_torn_monthly.hdf', 'r') as h5io:
#     clim = h5io['clim'][:]
    
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

lead_window, flag_shift = neighbour_leads(lead)

record_all = ()

for i, lead_temp in enumerate(lead_window):
    
    flag_ = flag_shift[i]
    
    with h5py.File(save_dir_scratch+'SPC_all_lead{}.hdf'.format(lead_temp), 'r') as h5io:
        record_temp = h5io['record_v3'][...]
        
    if flag_shift[i] == 0:
        record_all = record_all + (record_temp,)
        
    if flag_shift[i] == -1:
        record_temp[1:, ...] = record_temp[:-1, ...]
        record_temp[0, ...] = np.nan
        record_all = record_all + (record_temp,)
    
    if flag_shift[i] == +1:
        record_temp[:-1, ...] = record_temp[1:, ...]
        record_temp[-1, ...] = np.nan
        record_all = record_all + (record_temp,)


shape_record = record_temp.shape      

record_v3 = np.concatenate(record_all, axis=1)

grid_shape = record_v3.shape

L_vars = len(ind_pick)

target_size = 24
input_size = 128

half_margin = int((input_size - target_size) / 2)

grid_shape_input = (1059, 1799)

batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch_aug/'
prefix = 'aug{}_day{:03d}_pos_indx{}_indy{}_lead{}.npy'

L_train = 500 #grid_shape[0]
#L_valid = grid_shape[0]-L_train


out_slice = np.empty((1, input_size, input_size, L_vars))

base_v3_s = datetime(2018, 7, 15)
base_v3_e = datetime(2020, 12, 2)

base_v4_s = datetime(2020, 12, 3)
base_v4_e = datetime(2022, 7, 15)

base_ref = datetime(2010, 1, 1)

date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+180)]

for f in range(int(grid_shape[-1]/2)):
    for i in range(500):
        
        date_time = date_list_v3[i]
        mon_ind = date_time.month - 1
        
        day = i
        
        lon_temp = record_v3[i, 2*f]
        lat_temp = record_v3[i, 2*f+1]
        #mag_temp = record_v3[i, 3*f+2]
        
        flag_obs = lon_temp + lat_temp

        if np.logical_not(np.isnan(flag_obs)):

            indx_3km, indy_3km = du.grid_search(lon_3km, lat_3km, np.array(lon_temp)[None], np.array(lat_temp)[None])
            indx_3km = indx_3km[0]
            indy_3km = indy_3km[0]
            
            count = 0
            for augx in range(0, target_size, 8):
                for augy in range(0, target_size, 8):

                    x_margin_left = augx
                    x_margin_right = target_size - augx

                    y_margin_bottom = augy
                    y_margin_top = target_size - augy

                    x_edge_left = indx_3km - x_margin_left - half_margin
                    x_edge_right = indx_3km + x_margin_right + half_margin

                    y_edge_bottom = indy_3km - y_margin_bottom - half_margin
                    y_edge_top = indy_3km + y_margin_top + half_margin

                    if x_edge_left >= 0 and y_edge_bottom >= 0 and x_edge_right <= grid_shape_input[0] and y_edge_top <= grid_shape_input[1]:
                        
                        if land_mask_3km[x_edge_left, y_edge_bottom] and land_mask_3km[x_edge_left, y_edge_top]:
                    
                            if land_mask_3km[x_edge_right, y_edge_bottom] and land_mask_3km[x_edge_right, y_edge_top]:
                        
                                hrrr_3km = HRRRv3_lead[day, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]
                                uh_check = hrrr_3km[32:-32, 32:-32, 3]

                                # if np.sum(uh_check>0.0) > 10:

                                for v, ind_var in enumerate(ind_pick):

                                    temp = hrrr_3km[..., ind_var]

                                    if ind_var == 0:
                                        temp[temp<0] = 0

                                    if log_norm[ind_var]:
                                        temp = np.log(np.abs(temp)+1)
                                    else:
                                        temp = (temp - means[ind_var])/stds[ind_var]

                                    out_slice[..., v] = temp

                                if np.sum(np.isnan(out_slice)) > 0:
                                    print('HRRR contains NaN')
                                    continue;
                                else:
                                    save_name = batch_dir+prefix.format(count, day, indx_3km, indy_3km, lead)
                                    print(save_name)
                                    np.save(save_name, out_slice)
                                    count += 1

                            else:
                                continue;


