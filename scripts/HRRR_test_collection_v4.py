import sys
from glob import glob

import h5py
import pygrib
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du


base_v2_s = datetime(2017, 1, 1)
base_v2_e = datetime(2018, 7, 11)
base_v4_s = datetime(2020, 12, 3)
base_v4_e = datetime(2022, 7, 13)

N_days_v4 = (base_v4_e-base_v4_s).days

dt_v4 = [base_v4_s + timedelta(days=d) for d in range(N_days_v4)]

month_select = [3, 4, 5, 6, 7, 8, 9, 10]

dt_v4_scope = []

for dt_temp in dt_v4:
    if dt_temp.month in month_select:
        dt_v4_scope.append(datetime.strftime(dt_temp, HRRR_dir+'fcst12hr/HRRR.%Y%m%d.natf12.grib2'))

with pygrib.open(dt_v4_scope[0]) as grbio:
    #var_list = grbio()
    var = grbio.select(name='Land-sea mask')[0]
    land_mask = var.values
    lat_hrrr, lon_hrrr = var.latlons()
    
    
var_list = ['10 metre U wind component',
            '10 metre V wind component',
            '2 metre temperature',
            'Surface pressure',
            'Convective available potential energy',
            'Convective inhibition',
            'U-component storm motion',
            'V-component storm motion',
            'Storm relative helicity',
            'Vertical u-component shear',
            'Vertical v-component shear',
            'Maximum/Composite radar reflectivity',
]

ind_list = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

N_vars = len(var_list)

save_name = ['u10', 'v10', 'T2m', 'slp', 'cape', 'cin', 'us', 'vs', 'heli', 'ushear', 'vshear', 'max_refl']

# dt_v4_scope = dt_v4_scope[:10]

L_v4 = len(dt_v4_scope)
grid_shape = lon_hrrr.shape
out_vars = np.empty((L_v4, N_vars)+grid_shape)

for i, filename in enumerate(dt_v4_scope):
    print(filename)
    with pygrib.open(filename) as grbio:
        
        for j, varname in enumerate(var_list):
            try:
                var = grbio.select(name=varname)[ind_list[j]]
                out_vars[i, j] = var.values
            except:
                print('missing')
                out_vars[i, j] = np.nan

tuple_save = (land_mask, lon_hrrr, lat_hrrr,
              out_vars[:, 0, ...], out_vars[:, 1, ...], out_vars[:, 2, ...], out_vars[:, 3, ...],
              out_vars[:, 4, ...], out_vars[:, 5, ...], out_vars[:, 6, ...], out_vars[:, 7, ...],
              out_vars[:, 8, ...], out_vars[:, 9, ...], out_vars[:, 10, ...], out_vars[:, 11, ...])

label_save = ['land_mask', 'lon', 'lat',] + save_name

du.save_hdf5(tuple_save, label_save, save_dir_scratch, 'HRRRv4_envi.hdf')

