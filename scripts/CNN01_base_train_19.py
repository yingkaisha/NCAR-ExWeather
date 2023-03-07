# general tools
import os
import sys
from glob import glob

# data tools
import time
import h5py
import random
import numpy as np
from random import shuffle

from datetime import datetime, timedelta
import re

# deep learning tools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import utils
from tensorflow.keras import Model

tf.config.run_functions_eagerly(True)

from keras_unet_collection import utils as k_utils

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

# ==================== #
weights_round = 1
save_round = 2
seeds = 777
# ==================== #

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
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
    
def create_model(input_shape=(64, 64, 15)):

    depths=[3, 3, 27, 3]
    projection_dims=[32, 64, 96, 128]
    drop_path_rate=0.0
    layer_scale_init_value=1e-6


    model_name='Branch64X'
    IN64 = layers.Input(shape=input_shape)
    X = IN64
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
        X_convnext = layers.Conv2D(filters=projection_dims[2], kernel_size=5, padding="same",
                                   groups=projection_dims[2], name="{}_down2_dconv{}".format(model_name, j))(X_convnext)
        X_convnext = layers.LayerNormalization(epsilon=1e-6, name="{}_down2_dconv{}_norm".format(model_name, j))(X_convnext)
        X_convnext = layers.Dense(4 * projection_dims[2], name="{}_down2_dense{}_p1".format(model_name, j))(X_convnext)
        X_convnext = layers.Activation("gelu", name="{}_down2_gelu{}".format(model_name, j))(X_convnext)
        X_convnext = layers.Dense(projection_dims[2], name="{}_down2_dense{}_p2".format(model_name, j))(X_convnext)

        X_convnext = LayerScale(layer_scale_init_value, projection_dims[2], name="{}_down2_layerscale{}".format(model_name, j))(X_convnext)

        X = X + X_convnext

    # ----- convnext block 3 ----- #

    X = layers.LayerNormalization(epsilon=1e-6, name="{}_down3_norm".format(model_name))(X)
    X = layers.Conv2D(projection_dims[3], kernel_size=2, padding='same', name="{}_down3".format(model_name))(X)

    for j in range(depths[3]):

        X_convnext = X
        X_convnext = layers.Conv2D(filters=projection_dims[3], kernel_size=5, padding="same",
                                   groups=projection_dims[3], name="{}_down3_dconv{}".format(model_name, j))(X_convnext)
        X_convnext = layers.LayerNormalization(epsilon=1e-6, name="{}_down3_dconv{}_norm".format(model_name, j))(X_convnext)
        X_convnext = layers.Dense(4 * projection_dims[3], name="{}_down3_dense{}_p1".format(model_name, j))(X_convnext)
        X_convnext = layers.Activation("gelu", name="{}_down3_gelu{}".format(model_name, j))(X_convnext)
        X_convnext = layers.Dense(projection_dims[3], name="{}_down3_dense{}_p2".format(model_name, j))(X_convnext)

        X_convnext = LayerScale(layer_scale_init_value, projection_dims[3], name="{}_down3_layerscale{}".format(model_name, j))(X_convnext)

        X = X + X_convnext

    V1 = X

    OUT = layers.GlobalMaxPooling2D(name="{}_head_pool64".format(model_name))(V1)
    model = Model(inputs=IN64, outputs=OUT, name=model_name)
    
    return model

def create_model_head():

    
    IN_vec = keras.Input((128,))    
    X = IN_vec
    #
    X = keras.layers.Dense(64)(X)
    X = keras.layers.Activation("relu")(X)
    X = keras.layers.BatchNormalization()(X)
    
    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=IN_vec, outputs=OUT)
    
    return model

def verif_metric(VALID_target, Y_pred):
    BS = np.mean((VALID_target.ravel() - Y_pred.ravel())**2)
    print('{}'.format(BS))
    metric = BS

    return metric

def name_extract(filenames):
    
    date_base = datetime(2020, 7, 14)
    date_base2 = datetime(2021, 1, 1)
    
    filename_train = []
    filename_valid = []
    
    base_v3_s = datetime(2018, 7, 15)
    base_v3_e = datetime(2020, 12, 2)

    base_v4_s = datetime(2020, 12, 3)
    base_v4_e = datetime(2022, 7, 15)

    base_ref = datetime(2010, 1, 1)
    
    date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
    date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+365+30)]
    
    base_ref = datetime(2019, 10, 1)
    date_list_v4x = [base_ref + timedelta(days=day) for day in range(429)]
    
    for i, name in enumerate(filenames):
        
        if 'v4x' in name:
            date_list = date_list_v4x
        elif 'v4' in name:
            date_list = date_list_v4
        else:
            date_list = date_list_v3
        
        nums = re.findall(r'\d+', name)
        day = int(nums[-4])
        day = date_list[day]
        
        if (day - date_base).days < 0:
            filename_train.append(name)
            
        else:
            if (day - date_base2).days < 0:
                filename_valid.append(name)

        
    return filename_train, filename_valid


ind_pick_from_batch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
L_vars = len(ind_pick_from_batch)

vers = ['v3', 'v4x', 'v4']
leads = [19, 20, 21, 22, 23]
filenames_pos = {}
filenames_neg = {}

for ver in vers:
    for lead in leads:
        if ver == 'v3':
            path_ = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v3/'
        elif ver == 'v4':
            path_ = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4_temp/'
        else:
            path_ = '/glade/campaign/cisl/aiml/ksha/NCAR_batch_v4x/'
            
        filenames_pos['{}_lead{}'.format(ver, lead)] = sorted(glob("{}*pos*lead{}.npy".format(path_, lead)))
        filenames_neg['{}_lead{}'.format(ver, lead)] = sorted(glob("{}*neg_neg_neg*lead{}.npy".format(path_, lead)))
        
        print('{}, lead{}, pos: {}, neg: {}'.format(ver, lead, 
                                                    len(filenames_pos['{}_lead{}'.format(ver, lead)]), 
                                                    len(filenames_neg['{}_lead{}'.format(ver, lead)])))
        
filenames_pos_train = {}
filenames_neg_train = {}

filenames_pos_valid = {}
filenames_neg_valid = {}

for ver in vers:
    for lead in leads:
        temp_namelist_pos = filenames_pos['{}_lead{}'.format(ver, lead)]
        temp_namelist_neg = filenames_neg['{}_lead{}'.format(ver, lead)]
        
        pos_train, pos_valid = name_extract(temp_namelist_pos)
        neg_train, neg_valid = name_extract(temp_namelist_neg)
        
        print('pos train: {} pos valid: {} neg train: {} neg valid {}'.format(len(pos_train), 
                                                                              len(pos_valid), 
                                                                              len(neg_train), 
                                                                              len(neg_valid)))
        
        filenames_pos_train['{}_lead{}'.format(ver, lead)] = pos_train
        filenames_neg_train['{}_lead{}'.format(ver, lead)] = neg_train
        
        filenames_pos_valid['{}_lead{}'.format(ver, lead)] = pos_valid
        filenames_neg_valid['{}_lead{}'.format(ver, lead)] = neg_valid
        
pos_train_all = []
neg_train_all = []
pos_valid_all = []
neg_valid_all = []

for ver in vers:
    for lead in leads:
        pos_train_all += filenames_pos_train['{}_lead{}'.format(ver, lead)]
        neg_train_all += filenames_neg_train['{}_lead{}'.format(ver, lead)]
        pos_valid_all += filenames_pos_valid['{}_lead{}'.format(ver, lead)]
        neg_valid_all += filenames_neg_valid['{}_lead{}'.format(ver, lead)]
        
filename_valid = neg_valid_all[::130] + pos_valid_all[::13]
print(len(filename_valid))
L_valid = len(filename_valid)

VALID_input_64 = np.empty((L_valid, 64, 64, L_vars))
VALID_target = np.ones(L_valid)

for i, name in enumerate(filename_valid):
    data = np.load(name)
    for k, c in enumerate(ind_pick_from_batch):
        
        VALID_input_64[i, ..., k] = data[..., c]

        if 'pos' in name:
            VALID_target[i] = 1.0
        else:
            VALID_target[i] = 0.0
            
save_dir = '/glade/work/ksha/NCAR/'
tuple_save = (VALID_input_64, VALID_target)
label_save = ['VALID_input_64', 'VALID_target']
du.save_hdf5(tuple_save, label_save, save_dir, 'CNN_Validation_lead19.hdf')


model_head = create_model_head()
model_base = create_model(input_shape=(64, 64, 15))

IN = layers.Input(shape=(64, 64, 15))

VEC = model_base(IN)
OUT = model_head(VEC)

model_final = Model(inputs=IN, outputs=OUT)

# ============================================= #
# Weights

if weights_round == 0:
    W_old = k_utils.dummy_loader('/glade/work/ksha/NCAR/Keras_models/RE2_peak_base5/')
else:
    W_old = k_utils.dummy_loader('/glade/work/ksha/NCAR/Keras_models/RE2_peak19_base{}/'.format(weights_round))
    
model_final.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer=keras.optimizers.Adam(lr=1e-4))
model_final.set_weights(W_old)

# ============================================= #
# Training loop

Y_pred = model_final.predict([VALID_input_64])
record_temp = verif_metric(VALID_target, Y_pred)

min_del = 0
max_tol = 1000 # early stopping with patience

epochs = 5000
batch_size = 200
L_train = 64 #int(len(TRAIN_Y_pick) / batch_size)

X_batch_64 = np.empty((batch_size, 64, 64, L_vars))
Y_batch = np.empty((batch_size, 1))

X_batch_64[...] = np.nan
Y_batch[...] = np.nan

temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

# =========== Model Section ========== #
key = 'RE2_peak19_base{}'.format(save_round)
model_name = '{}'.format(key)
model_path = temp_dir+model_name

tol = 0

filename_pos_train = pos_train_all
filename_neg_train = neg_train_all
# ========== Training loop ========== #
L_pos = len(filename_pos_train)
L_neg = len(filename_neg_train)

record = record_temp
print("Initial record: {}".format(record))

set_seeds(seeds)
    
for i in range(epochs):
    start_time = time.time()

    # loop of batch
    for j in range(L_train):
        N_pos = 20
        N_neg = batch_size - N_pos

        ind_neg = du.shuffle_ind(L_neg)
        ind_pos = du.shuffle_ind(L_pos)

        file_pick_neg = []
        for ind_temp in ind_neg[:N_neg]:
            file_pick_neg.append(filename_neg_train[ind_temp])

        file_pick_pos = []
        for ind_temp in ind_pos[:N_pos]:
            file_pick_pos.append(filename_pos_train[ind_temp])

        file_pick = file_pick_neg + file_pick_pos

        if len(file_pick) != batch_size:
            sregwet

        for k in range(batch_size):
            data = np.load(file_pick[k])

            for l, c in enumerate(ind_pick_from_batch):
                temp = data[..., c] 
                X_batch_64[k, ..., l] = temp

            if 'pos' in file_pick[k]:
                Y_batch[k, :] = 1.0 #np.random.uniform(0.9, 0.99)
            elif 'neg_neg_neg' in file_pick[k]:
                Y_batch[k, :] = 0.0 #np.random.uniform(0.01, 0.05)
            else:
                werhgaer

        ind_ = du.shuffle_ind(batch_size)
        X_batch_64 = X_batch_64[ind_, ...]
        Y_batch = Y_batch[ind_, :]

        # train on batch
        model_final.train_on_batch(X_batch_64, Y_batch);

    # epoch end operations
    Y_pred = model_final.predict([VALID_input_64])
    # Y_pred[Y_pred<0] = 0
    # Y_pred[Y_pred>1] = 1

    record_temp = verif_metric(VALID_target, Y_pred)

    # if i % 10 == 0:
    #     model.save(model_path_backup)

    if (record - record_temp > min_del):
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        tol = 0
        #print('tol: {}'.format(tol))
        # save
        print('save to: {}'.format(model_path))
        model_final.save(model_path)
    else:
        print('Validation loss {} NOT improved'.format(record_temp))
        if record_temp >= 2.0:
            print('Early stopping')
            break;
        else:
            tol += 1
            if tol >= max_tol:
                print('Early stopping')
                break;
            else:
                continue;
    print("--- %s seconds ---" % (time.time() - start_time))
