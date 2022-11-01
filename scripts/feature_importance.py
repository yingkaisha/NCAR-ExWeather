# general tools
import sys
from glob import glob

# data tools
import time
import h5py
import random
import numpy as np
from random import shuffle

# deep learning tools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.config.run_functions_eagerly(True)

from keras_unet_collection import models as k_models
from keras_unet_collection import utils as k_utils

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du
import graph_utils as gu
import convnext_keras as ck

from sklearn.metrics import classification_report, auc, roc_curve
from sklearn.metrics import confusion_matrix


def test_metric(VALID_target, Y_pred, thres=0.5):

    tn, fp, fn, tp = confusion_matrix(VALID_target.ravel(), Y_pred.ravel()>thres).ravel()

    return tn, fp, fn, tp

with h5py.File(save_dir_scratch+'VALID_real_lead{}.hdf'.format(21), 'r') as h5io:
    TEST_input = h5io['TEST_input'][...]
    TEST_target = h5io['TEST_target'][...]

TEST_target[np.isnan(TEST_target)] = 0.0
TEST_target[TEST_target!=0] = 1.0

TEST_input = TEST_input[...]
TEST_target = TEST_target[...]

MODEL_CONFIGS = {

    "small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
}}

model = ck.ConvNeXt(
        depths=MODEL_CONFIGS["small"]["depths"],
        projection_dims=MODEL_CONFIGS["small"]["projection_dims"],
        drop_path_rate=0.0,
        layer_scale_init_value=0.1,
        model_name='test',
        input_shape=(128, 128, 19),
        pooling='max',
        classes=1,)#0.5 #0.1 #1e-6

batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch/'
temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

W_old = k_utils.dummy_loader(temp_dir+'CONVNEXT_Base_pp19_tune3')
model.set_weights(W_old)

model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.SGD(lr=0))

L_valid = len(TEST_input)

rounds = 1
N_var = 19
AUC_drop = np.empty((N_var, 4, 1))

for r in range(rounds):
    for n in range(N_var):
        start_time = time.time()
        print('shuffling var {}'.format(n))

        ind_ = du.shuffle_ind(L_valid)
        TEST_input_shuffle = np.copy(TEST_input)
        TEST_input_shuffle[..., n] = TEST_input_shuffle[ind_, ..., n]

        Y_pred = model.predict([TEST_input_shuffle,])
        
        Y_pred[Y_pred<0.1] = 0
        Y_pred[Y_pred>1] = 1
        
        AUC_drop[n, :, r] = test_metric(TEST_target, Y_pred, thres=0.5)
        print("--- %s seconds ---" % (time.time() - start_time))

np.save(temp_dir+'CONVNEXT_Base_pp19_tune3'+'_FI_2.npy', AUC_drop)
