
import sys
from glob import glob

import zarr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

base_v3_s = datetime(2018, 7, 15)
base_v3_e = datetime(2020, 12, 2)

base_v4_s = datetime(2020, 12, 3)
base_v4_e = datetime(2022, 7, 15)

base_ref = datetime(2010, 1, 1)

ref_v3 = (base_v3_e - base_ref).days

var_inds = [1, 46, 64, 76, 88, 106, 181]
filenames = sorted(glob(data_pdhrrr_dir+'*.par')) #[:20]

N_var = len(var_inds)
N_grids = 1308
N_leads = 18
N_files = len(filenames)
N_samples = N_files*N_grids
N_cut = N_grids*N_leads

months = [2, 3, 4, 5, 6, 7, 8, 9, 10]

data_pos_v3 = np.empty((N_samples, N_leads, N_var))
data_neg_v3 = np.empty((N_samples, N_leads, N_var))
data_pos_v4 = np.empty((N_samples, N_leads, N_var))
data_neg_v4 = np.empty((N_samples, N_leads, N_var))

count_pos_v3 = 0
count_neg_v3 = 0
count_pos_v4 = 0
count_neg_v4 = 0

for i, filename in enumerate(filenames):
    
    print(filename)
    
    dt_sring = filename[-19:-11]
    dt_ = datetime.strptime(dt_sring, '%Y%m%d')
    flag_v3 = (dt_ - base_ref).days < ref_v3
    
    if dt_.month in months is False:
        continue;
    else:
        dataframe = pd.read_parquet(filename, engine='pyarrow')
    
    if len(dataframe) < N_cut:
        continue;
    else:
        data = dataframe.values[:N_cut, var_inds]
    
    if flag_v3:
        for n in range(N_grids-1):
            data_slice = data[n:-1:1308, :]

            if np.max(data_slice[:, -1])>0 and np.max(data_slice[:, -1])<= 4000:
                data_pos_v3[count_pos_v3, ...] = data_slice
                count_pos_v3 += 1
            else:
                data_neg_v3[count_neg_v3, ...] = data_slice
                count_neg_v3 += 1
                
    else:
        for n in range(N_grids-1):
            data_slice = data[n:-1:1308, :]

            if np.max(data_slice[:, -1])>0 and np.max(data_slice[:, -1])<= 4000:
                data_pos_v4[count_pos_v4, ...] = data_slice
                count_pos_v4 += 1
            else:
                data_neg_v4[count_neg_v4, ...] = data_slice
                count_neg_v4 += 1
                
data_pos_v3 = data_pos_v3[:count_pos_v3, ...]
data_neg_v3 = data_neg_v3[:count_neg_v3, ...]

data_pos_v4 = data_pos_v4[:count_pos_v4, ...]
data_neg_v4 = data_neg_v4[:count_neg_v4, ...]

zarr.save(save_dir_scratch+'HRRR_clean_pos_v3.zarr', data_pos_v3)
zarr.save(save_dir_scratch+'HRRR_clean_neg_v3.zarr', data_neg_v3)

zarr.save(save_dir_scratch+'HRRR_clean_pos_v4.zarr', data_pos_v4)
zarr.save(save_dir_scratch+'HRRR_clean_neg_v4.zarr', data_neg_v4)                
