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

lead = 12

start_time = time.time()

HRRRv3_lead = zarr.load(save_dir_scratch+'HRRR_{}_v3.zarr'.format(lead))

print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

for i in range(23):

    temp_var = HRRRv3_lead[..., i].ravel()

    temp_var_clean = temp_var[~np.isnan(temp_var)]

    temp_min = np.min(temp_var_clean)
    temp_max = np.max(temp_var_clean)
    temp_mean = np.mean(temp_var_clean)
    temp_std = np.std(temp_var_clean)
    temp_90 = np.quantile(temp_var_clean, 0.90)
    temp_95 = np.quantile(temp_var_clean, 0.95)

    print('Max: {}, Min: {}, Mean: {}, Std: {}\n90th: {}, 95th: {}'.format(temp_max, temp_min, temp_mean, temp_std, temp_90, temp_95))

