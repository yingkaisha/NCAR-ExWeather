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
from keras_unet_collection import utils as k_utils
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

# ---------------------- #
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import Model


class LayerScale(layers.Layer):
    """Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.
    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config
    
depths=[3, 3, 27, 3]
projection_dims=[96, 192, 384, 768]
drop_path_rate=0.0
layer_scale_init_value=0.5
model_name='test'
input_shape=(128, 128, 4)
classes=1

IN = layers.Input(shape=input_shape)
X = IN

num_convnext_blocks = 4

# ----- convnext block 0 ----- #

X = layers.Conv2D(projection_dims[0], kernel_size=4, strides=4, name="{}_down0".format(model_name))(X)
X = layers.LayerNormalization(epsilon=1e-6, name="{}_down0_norm".format(model_name))(X)

for j in range(depths[0]):
    
    X_convnext = X
    X_convnext = layers.Conv2D(filters=projection_dims[0], kernel_size=7, padding="same",
                               groups=projection_dims[0], name="{}_down0_dconv{}".format(model_name, j))(X_convnext)
    X_convnext = layers.LayerNormalization(epsilon=1e-6, name="{}_down0_dconv{}_norm".format(model_name, j))(X_convnext)
    X_convnext = layers.Dense(4 * projection_dims[0], name="{}_down0_dense{}_p1".format(model_name, j))(X_convnext)
    X_convnext = layers.Activation("gelu", name="{}_down0_gelu{}".format(model_name, j))(X_convnext)
    X_convnext = layers.Dense(projection_dims[0], name="{}_down0_dense{}_p2".format(model_name, j))(X_convnext)

    X_convnext = LayerScale(layer_scale_init_value, projection_dims[0], name="{}_down0_layerscale{}".format(model_name, j))(X_convnext)

    X = X + X_convnext


# ----- convnext block 1 ----- #

X = layers.LayerNormalization(epsilon=1e-6, name="{}_down1_norm".format(model_name))(X)
X = layers.Conv2D(projection_dims[1], kernel_size=2, strides=2, name="{}_down1".format(model_name))(X)

for j in range(depths[1]):
    
    X_convnext = X
    X_convnext = layers.Conv2D(filters=projection_dims[1], kernel_size=7, padding="same",
                               groups=projection_dims[1], name="{}_down1_dconv{}".format(model_name, j))(X_convnext)
    X_convnext = layers.LayerNormalization(epsilon=1e-6, name="{}_down1_dconv{}_norm".format(model_name, j))(X_convnext)
    X_convnext = layers.Dense(4 * projection_dims[1], name="{}_down1_dense{}_p1".format(model_name, j))(X_convnext)
    X_convnext = layers.Activation("gelu", name="{}_down1_gelu{}".format(model_name, j))(X_convnext)
    X_convnext = layers.Dense(projection_dims[1], name="{}_down1_dense{}_p2".format(model_name, j))(X_convnext)

    X_convnext = LayerScale(layer_scale_init_value, projection_dims[1], name="{}_down1_layerscale{}".format(model_name, j))(X_convnext)

    X = X + X_convnext

# ----- convnext block 2 ----- #

X = layers.LayerNormalization(epsilon=1e-6, name="{}_down2_norm".format(model_name))(X)
X = layers.Conv2D(projection_dims[2], kernel_size=2, strides=2, name="{}_down2".format(model_name))(X)

for j in range(depths[2]):
    
    X_convnext = X
    X_convnext = layers.Conv2D(filters=projection_dims[2], kernel_size=7, padding="same",
                               groups=projection_dims[2], name="{}_down2_dconv{}".format(model_name, j))(X_convnext)
    X_convnext = layers.LayerNormalization(epsilon=1e-6, name="{}_down2_dconv{}_norm".format(model_name, j))(X_convnext)
    X_convnext = layers.Dense(4 * projection_dims[2], name="{}_down2_dense{}_p1".format(model_name, j))(X_convnext)
    X_convnext = layers.Activation("gelu", name="{}_down2_gelu{}".format(model_name, j))(X_convnext)
    X_convnext = layers.Dense(projection_dims[2], name="{}_down2_dense{}_p2".format(model_name, j))(X_convnext)

    X_convnext = LayerScale(layer_scale_init_value, projection_dims[2], name="{}_down2_layerscale{}".format(model_name, j))(X_convnext)

    X = X + X_convnext

# ----- convnext block 3 ----- #

X = layers.LayerNormalization(epsilon=1e-6, name="{}_down3_norm".format(model_name))(X)
X = layers.Conv2D(projection_dims[3], kernel_size=2, strides=1, name="{}_down3".format(model_name))(X)

for j in range(depths[3]):
    
    X_convnext = X
    X_convnext = layers.Conv2D(filters=projection_dims[3], kernel_size=7, padding="same",
                               groups=projection_dims[3], name="{}_down3_dconv{}".format(model_name, j))(X_convnext)
    X_convnext = layers.LayerNormalization(epsilon=1e-6, name="{}_down3_dconv{}_norm".format(model_name, j))(X_convnext)
    X_convnext = layers.Dense(4 * projection_dims[3], name="{}_down3_dense{}_p1".format(model_name, j))(X_convnext)
    X_convnext = layers.Activation("gelu", name="{}_down3_gelu{}".format(model_name, j))(X_convnext)
    X_convnext = layers.Dense(projection_dims[3], name="{}_down3_dense{}_p2".format(model_name, j))(X_convnext)

    X_convnext = LayerScale(layer_scale_init_value, projection_dims[3], name="{}_down3_layerscale{}".format(model_name, j))(X_convnext)

    X = X + X_convnext

OUT = X

OUT = layers.GlobalMaxPooling2D(name="{}_head_pool".format(model_name))(OUT)
OUT = layers.LayerNormalization(epsilon=1e-6, name="{}_head_pool_norm".format(model_name))(OUT)

OUT = layers.Dense(256, name="{}_head_dense1".format(model_name))(OUT)
OUT = layers.Activation("gelu", name="{}_head_dense1_gelu".format(model_name))(OUT)
OUT = layers.LayerNormalization(epsilon=1e-6, name="{}_head_dense1_norm".format(model_name))(OUT)

OUT = layers.Dense(64, name="{}_head_dense2".format(model_name))(OUT)
OUT = layers.Activation("gelu", name="{}_head_dense2_gelu".format(model_name))(OUT)
OUT = layers.LayerNormalization(epsilon=1e-6, name="{}_head_dense2_norm".format(model_name))(OUT)

#OUT = layers.Dense(1, name="{}_head_out".format(model_name))(OUT)

model = Model(inputs=IN, outputs=OUT, name=model_name)

batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch/'
temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

W_new = model.get_weights()
W_old = k_utils.dummy_loader(temp_dir+'NEW_D_pp4_tune3')

for l in range(len(W_new)):
    if W_new[l].shape == W_old[l].shape:
        W_new[l] = W_old[l]

model.set_weights(W_new)


names = [
    '0 Max/Comp Radar',
    '1 MSLP',
    '2 AGL',
    '3 UH 2-5 km',
    '4 UH 0-2 km',
    '5 UH 0-3 km',
    '6 Vorticity 0-2 km',
    '7 Vorticity 0-1 km',
    '8 Graupel mass',
    '9 T 2m',
    '10 Dewpoint 2m',
    '11 U 10m',
    '12 V 10m',
    '13 SPD 10m',
    '14 APCP',
    '15 CAPE',
    '16 CIN',
    '17 SRH 0-3 km',
    '18 SRH 0-1 km',
    '19 U shear 0-1 km',
    '20 V shear 0-1 km',
    '21 U shear 0-6 km',
    '22 V shear 0-6 km']

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

log_norm = [True, False, True, True, True, True, True, True, True, False, False, 
            False, False, True, True, True, True, False, False, False, False, False, False]


ind_pick = [3, 15, 21, 22]
rad = 4

indx_loc = 20
indy_loc = 45

input_size = 128
half_margin = 64

with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_72km = h5io['lon_72km'][...]
    lat_72km = h5io['lat_72km'][...]
    land_mask_72km = h5io['land_mask_72km'][...]
    land_mask_3km = h5io['land_mask_3km'][...]
    
with h5py.File('/glade/scratch/ksha/DRIVE/SPC_72km_all.hdf', 'r') as h5io:
    record_merge = h5io['record_v3'][...]
    
shape_72km = lon_72km.shape
shape_3km = lon_3km.shape

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
        
indx_min = int(indx_array.min())
indx_max = int(indx_array.max())

indy_min = int(indy_array.min())
indy_max = int(indy_array.max())

HRRRv3_lead_n3 = da.from_zarr(save_dir_scratch+'HRRR_{:02}_v3.zarr'.format(lead-3))
HRRRv3_lead_n2 = da.from_zarr(save_dir_scratch+'HRRR_{:02}_v3.zarr'.format(lead-2))
HRRRv3_lead_n1 = da.from_zarr(save_dir_scratch+'HRRR_{:02}_v3.zarr'.format(lead-1))
HRRRv3_lead_p0 = da.from_zarr(save_dir_scratch+'HRRR_{:02}_v3.zarr'.format(lead))
HRRRv3_lead_p1 = da.from_zarr(save_dir_scratch+'HRRR_{:02}_v3.zarr'.format(lead+1))
HRRRv3_lead_p2 = da.from_zarr(save_dir_scratch+'HRRR_{:02}_v3.zarr'.format(lead+2))

HRRRv3_lead_local_n3 = HRRRv3_lead_n3[..., ind_pick]
HRRRv3_lead_local_n2 = HRRRv3_lead_n2[..., ind_pick]
HRRRv3_lead_local_n1 = HRRRv3_lead_n1[..., ind_pick]
HRRRv3_lead_local_p0 = HRRRv3_lead_p0[..., ind_pick]
HRRRv3_lead_local_p1 = HRRRv3_lead_p1[..., ind_pick]
HRRRv3_lead_local_p2 = HRRRv3_lead_p2[..., ind_pick]

shape_3km = HRRRv3_lead_local_p0.shape[1:3]

base_v3_s = datetime(2018, 7, 15)
base_v3_e = datetime(2020, 12, 2)

base_v4_s = datetime(2020, 12, 3)
base_v4_e = datetime(2022, 7, 15)

base_ref = datetime(2010, 1, 1)

date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+180)]

L_train = len(date_list_v3)

input_size = 128
half_margin = 64

L_vars = len(ind_pick)

out_slice_n3 = np.empty((1, input_size, input_size, L_vars))
out_slice_n2 = np.empty((1, input_size, input_size, L_vars))
out_slice_n1 = np.empty((1, input_size, input_size, L_vars))
out_slice_p0 = np.empty((1, input_size, input_size, L_vars))
out_slice_p1 = np.empty((1, input_size, input_size, L_vars))
out_slice_p2 = np.empty((1, input_size, input_size, L_vars))

batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch2/'
prefix = '{}_day{:03d}_{}_{}_{}_indx{}_indy{}_lead{}.npy'

flag_torn = 'neg'
flag_wind = 'neg'
flag_hail = 'neg'

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
                        
#L_train
for day in range(3, L_train):
    if day > 600:
        tv_label = 'VALID'
    else:
        tv_label = 'TRAIN'
        
    # if np.nansum(record_v3[day, ...]) == 0:
    #     continue;
    
    for ix in range(indx_loc-rad, indx_loc+rad+1, 1):
        for iy in range(indy_loc-rad, indy_loc+rad+1, 1):
            
            indx = int(indx_array[ix, iy])
            indy = int(indy_array[ix, iy])
            
            x_edge_left = indx - half_margin
            x_edge_right = indx + half_margin

            y_edge_bottom = indy - half_margin
            y_edge_top = indy + half_margin
            
            if x_edge_left >= 0 and y_edge_bottom >= 0 and x_edge_right < shape_3km[0] and y_edge_top < shape_3km[1]:

                if land_mask_3km[x_edge_left, y_edge_bottom] and land_mask_3km[x_edge_left, y_edge_top]:
                    
                    if land_mask_3km[x_edge_right, y_edge_bottom] and land_mask_3km[x_edge_right, y_edge_top]:
                        hrrr_n3 = HRRRv3_lead_n3[day, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]
                        hrrr_n2 = HRRRv3_lead_n2[day, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]
                        hrrr_n1 = HRRRv3_lead_n1[day, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]
                        hrrr_p0 = HRRRv3_lead_p0[day, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]
                        hrrr_p1 = HRRRv3_lead_p1[day, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]
                        hrrr_p2 = HRRRv3_lead_p2[day, x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]


                        for v, ind_var in enumerate(ind_pick):
                            temp_n3 = hrrr_n3[..., v]
                            temp_n2 = hrrr_n2[..., v]
                            temp_n1 = hrrr_n1[..., v]
                            temp_p0 = hrrr_p0[..., v]
                            temp_p1 = hrrr_p1[..., v]
                            temp_p2 = hrrr_p2[..., v]

                            if log_norm[ind_var]:
                                temp_n3 = np.log(np.abs(temp_n3)+1)
                                temp_n2 = np.log(np.abs(temp_n2)+1)
                                temp_n1 = np.log(np.abs(temp_n1)+1)
                                temp_p0 = np.log(np.abs(temp_p0)+1)
                                temp_p1 = np.log(np.abs(temp_p1)+1)
                                temp_p2 = np.log(np.abs(temp_p2)+1)
                            else:
                                temp_n3 = (temp_n3 - means[ind_var])/stds[ind_var]
                                temp_n2 = (temp_n2 - means[ind_var])/stds[ind_var]
                                temp_n1 = (temp_n1 - means[ind_var])/stds[ind_var]
                                temp_p0 = (temp_p0 - means[ind_var])/stds[ind_var]
                                temp_p1 = (temp_p1 - means[ind_var])/stds[ind_var]
                                temp_p2 = (temp_p2 - means[ind_var])/stds[ind_var]

                            out_slice_n3[..., v] = temp_n3
                            out_slice_n2[..., v] = temp_n2
                            out_slice_n1[..., v] = temp_n1
                            out_slice_p0[..., v] = temp_p0
                            out_slice_p1[..., v] = temp_p1
                            out_slice_p2[..., v] = temp_p2
                            
                        obs_temp = record_v3[day, ix, iy, :]
                        
                        obs_history = np.concatenate((record_merge[day-3, ix, iy, :, lead:], 
                                                      record_merge[day-2, ix, iy, :, :], 
                                                      record_merge[day-1, ix, iy, :, :lead]), axis=-1)
                        obs_history = np.max(obs_history, axis=0)
                        
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
                        N_nan = np.sum(np.isnan(out_slice_n3)) + np.sum(np.isnan(out_slice_n2)) + np.sum(np.isnan(out_slice_n1)) + np.sum(np.isnan(out_slice_p0)) + np.sum(np.isnan(out_slice_p1)) + np.sum(np.isnan(out_slice_p2))
                        if N_nan > 0:
                            print('HRRR contains NaN')
                            continue;
                        else:
                            save_name = batch_dir+prefix.format(tv_label, day, flag_torn, flag_wind, flag_hail, ix, iy, lead)
                            print(save_name)
                            
                            save_dict = {}
                            save_dict['Gn3'] = model.predict([out_slice_n3,])
                            save_dict['Gn2'] = model.predict([out_slice_n2,])
                            save_dict['Gn1'] = model.predict([out_slice_n1,])
                            save_dict['Gp0'] = model.predict([out_slice_p0,])
                            save_dict['Gp1'] = model.predict([out_slice_p1,])
                            save_dict['Gp2'] = model.predict([out_slice_p2,])
                            save_dict['code'] = obs_history
                            
                            np.save(save_name, save_dict)

                        