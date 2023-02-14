
import sys
from glob import glob

import time
import h5py
import zarr
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

from datetime import datetime, timedelta

import dask.array as da

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

ind_pick = [0, 1, 3, 4, 8, 9, 10, 13, 14, 15, 16, 17, 18, 21, 22] 
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

log_norm = [True, False, True, True, True, True, True, True, True, False, False, 
            False, False, True, True, True, True, False, False, False, False, False, False]

sparse = [True, False, True, True, True, True, True, True, True, False, False, 
          False, False, False, True, True, True, False, False, False, False, False, False]

HRRRv3_lead = zarr.load(save_dir_scratch+'HRRR_{:02}_v3.zarr'.format(lead))
# HRRRv3_lead = np.zeros((872, 1059, 1799, 23)) # use fake data for a quick test

lead_window, flag_shift = neighbour_leads(lead)

record_all = ()

for i, lead_temp in enumerate(lead_window):
    
    flag_ = flag_shift[i]
    
    with h5py.File(save_dir_scratch+'SPC_to_lead{}_72km_all.hdf'.format(lead_temp), 'r') as h5io:
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
record_v3 = np.empty(shape_record)
record_v3[...] = np.nan

for i in range(4):
    record_temp = record_all[i]
    for day in range(shape_record[0]):
        for ix in range(shape_record[1]):
            for iy in range(shape_record[2]):
                for event in range(shape_record[3]):
                    if np.logical_not(np.isnan(record_temp[day, ix, iy, event])):
                        record_v3[day, ix, iy, event] = record_temp[day, ix, iy, event]
        

with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_72km = h5io['lon_80km'][...]
    lat_72km = h5io['lat_80km'][...]
    land_mask_72km = h5io['land_mask_80km'][...]
    land_mask_3km = h5io['land_mask_3km'][...]
    
shape_72km = lon_72km.shape
shape_3km = lon_3km.shape
shape_record = record_v3.shape

indx_array = np.empty(shape_72km)
indy_array = np.empty(shape_72km)

gridTree = cKDTree(list(zip(lon_3km.ravel(), lat_3km.ravel()))) #KDTree_wraper(xgrid, ygrid)

for xi in range(shape_72km[0]):
    for yi in range(shape_72km[1]):
        
        temp_lon = lon_72km[xi, yi]
        temp_lat = lat_72km[xi, yi]
        
        dist, indexes = gridTree.query(list(zip(np.array(temp_lon)[None], np.array(temp_lat)[None])))
        indx_3km, indy_3km = np.unravel_index(indexes, shape_3km)
        
        indx_array[xi, yi] = indx_3km[0]
        indy_array[xi, yi] = indy_3km[0]
        
base_v3_s = datetime(2018, 7, 15)
base_v3_e = datetime(2020, 12, 2)

base_v4_s = datetime(2020, 12, 3)
base_v4_e = datetime(2022, 7, 15)

base_ref = datetime(2010, 1, 1)

date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+180)]

L_train = shape_record[0]

input_size = 64
half_margin = 32

L_vars = len(ind_pick)
L_vars_per = len(ind_pick)

out_slice = np.empty((1, input_size, input_size, L_vars))

batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch_raw/'
prefix = '{}_day{:03d}_{}_{}_{}_indx{}_indy{}_lead{}.npy'

flag_torn = 'neg'
flag_wind = 'neg'
flag_hail = 'neg'

for day in range(L_train):
    if day > 600:
        tv_label = 'VALID'
    else:
        tv_label = 'TRAIN'
        
    # if np.nansum(record_v3[day, ...]) == 0:
    #     continue;
    
    for ix in range(shape_72km[0]):
        for iy in range(shape_72km[1]):
            
            indx = int(indx_array[ix, iy])
            indy = int(indy_array[ix, iy])
            
            x_edge_left = indx - half_margin
            x_edge_right = indx + half_margin

            y_edge_bottom = indy - half_margin
            y_edge_top = indy + half_margin
            
            if x_edge_left >= 0 and y_edge_bottom >= 0 and x_edge_right < shape_3km[0] and y_edge_top < shape_3km[1]:

                if land_mask_3km[x_edge_left, y_edge_bottom] and land_mask_3km[x_edge_left, y_edge_top]:
                    
                    if land_mask_3km[x_edge_right, y_edge_bottom] and land_mask_3km[x_edge_right, y_edge_top]:
                
                        hrrr_3km = HRRRv3_lead[day, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]

                        for v, ind_var in enumerate(ind_pick):

                            temp = hrrr_3km[..., ind_var]
                            if ind_var == 0:
                                temp[temp<0] = 0
                            if log_norm[ind_var]:
                                temp = np.log(np.abs(temp)+1)
                                
                            max_ = np.max(temp)
                            min_ = np.min(temp)
                            temp = (temp - min_) / (max_ - min_)
                            
                            out_slice[..., v] = temp

                        obs_temp = record_v3[day, ix, iy, :]

                        if obs_temp[0] == 0:
                            flag_torn = 'neg'
                        else:
                            flag_torn = 'pos'

                        if obs_temp[1] == 0:
                            flag_wind = 'neg'
                        else:
                            flag_wind = 'pos'

                        if obs_temp[2] == 0:
                            flag_hail = 'neg'
                        else:
                            flag_hail = 'pos' 

                        if np.sum(np.isnan(out_slice)) > 0:
                            print('HRRR contains NaN')
                            continue;
                        else:
                            save_name = batch_dir+prefix.format(tv_label, day, flag_torn, flag_wind, flag_hail, ix, iy, lead)
                            print(save_name)
                            np.save(save_name, out_slice)

