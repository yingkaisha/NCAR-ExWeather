
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

from sklearn.metrics import classification_report, auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss

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
    
def Head(num_classes=1000, name=None):
    """Implementation of classification head of RegNet.
    Args:
      num_classes: number of classes for Dense layer
      name: name prefix
    Returns:
      Classification head function.
    """
    if name is None:
        name = str(backend.get_uid("head"))

    def apply(x):
        x = layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
        x = layers.LayerNormalization(
            epsilon=1e-6, name=name + "_head_layernorm"
        )(x)
        x = layers.Dense(num_classes, name=name + "_head_dense")(x)
        return x

    return apply

def create_model(input_shape=(64, 64, 14)):

    depths=[3, 3, 27, 3]
    projection_dims=[96, 192, 384, 768]
    drop_path_rate=0.0
    layer_scale_init_value=1e-6


    model_name='Branch64X'
    IN64 = layers.Input(shape=input_shape)
    X = IN64

    X = layers.LocallyConnected2D(64, kernel_size=1, strides=(1, 1), padding="valid", implementation=1)(X)
    X = layers.LayerNormalization(epsilon=1e-6, name="{}_lc1_norm".format(model_name))(X)
    X = layers.Activation("gelu", name="{}_lc1_gelu".format(model_name))(X)

    # X = layers.LocallyConnected2D(96, kernel_size=1, strides=(1, 1), padding="valid", implementation=1)(X)
    # X = layers.LayerNormalization(epsilon=1e-6, name="{}_lc2_norm".format(model_name))(X)
    # X = layers.Activation("gelu", name="{}_lc2_gelu".format(model_name))(X)

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
    OUT = layers.LayerNormalization(epsilon=1e-6, name="{}_head_norm64".format(model_name))(OUT)

    OUT = layers.Dense(256, name="{}_dense1".format(model_name))(OUT)
    OUT = layers.LayerNormalization(epsilon=1e-6, name="{}_dense1_norm".format(model_name))(OUT)
    OUT = layers.Activation("gelu", name="{}_dense1_gelu{}".format(model_name, j))(OUT)

    OUT = layers.Dense(1, name="{}_head_out".format(model_name))(OUT)

    model = Model(inputs=IN64, outputs=OUT, name=model_name)

    return model

def verif_metric(VALID_target, Y_pred):


    # fpr, tpr, thresholds = roc_curve(VALID_target.ravel(), Y_pred.ravel())
    # AUC = auc(fpr, tpr)
    # AUC_metric = 1 - AUC
    
    BS = np.mean((VALID_target.ravel() - Y_pred.ravel())**2)
    #ll = log_loss(VALID_target.ravel(), Y_pred.ravel())
    
    print('{}'.format(BS))
    metric = BS

    return metric


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('try', help='try')
args = vars(parser.parse_args())

# =============== #
try_ = int(args['try'])

with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_72km = h5io['lon_72km'][...]
    lat_72km = h5io['lat_72km'][...]
    land_mask_72km = h5io['land_mask_72km'][...]
    land_mask_3km = h5io['land_mask_3km'][...]

ind_pick_from_batch = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
L_vars = len(ind_pick_from_batch)

filename_neg_train = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead2.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead3.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead4.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead5.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead6.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead7.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead8.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead9.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead10.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead11.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*neg_neg_neg*lead12.npy"))

filename_pos_train = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead2.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead3.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead4.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead5.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead6.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead7.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead8.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead9.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead10.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead11.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/*pos*lead12.npy"))

filename_neg_valid = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead2.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead3.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead4.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead5.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead6.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead7.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead8.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead9.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead10.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead11.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*neg_neg_neg*lead12.npy"))

filename_pos_valid = sorted(glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead2.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead3.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead4.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead5.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead6.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead7.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead8.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead9.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead10.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead11.npy")+\
                            glob("/glade/campaign/cisl/aiml/ksha/NCAR_batch/VALID*pos*lead12.npy"))

filename_valid = filename_neg_valid[::500] + filename_pos_valid
#filename_valid = filename_neg_valid[::5000] + filename_pos_valid[::10]

L_valid = len(filename_valid)
L_var = L_vars

TEST_input_64 = np.empty((L_valid, 64, 64, L_var))
TEST_target = np.ones(L_valid)

for i, name in enumerate(filename_valid):
    data = np.load(name)
    for k, c in enumerate(ind_pick_from_batch):
        
        TEST_input_64[i, ..., k] = data[..., c]

        if 'pos' in name:
            TEST_target[i] = 1.0
        else:
            TEST_target[i] = 0.0

            
# ========== Training hyper-params ========== #

training_rounds = 20

seeds_base = [12342, 2536234, 98765, 473, 865, 7456, 69472, 3456357, 3425, 678,
         2452624, 5787, 235362, 67896, 98454, 12445, 46767, 78906, 345, 8695]

seeds = list(np.array(seeds_base)+try_)

min_del = 0
max_tol = 5 # early stopping with patience

epochs = 500
batch_size = 200
L_train = 100

# ========== model name and file path ========== #

batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch/'
temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

key = 'SK14'
model_name = '{}_pp14_try{}_tune2'.format(key, try_)
model_path = temp_dir+model_name


# ========== Allocation ========== #

X_batch_64 = np.empty((batch_size, 64, 64, L_vars))
Y_batch = np.empty((batch_size, 1))

X_batch_64[...] = np.nan
Y_batch[...] = np.nan

# ========== Training loop ========== #

tol = 0

L_pos = len(filename_pos_train)
L_neg = len(filename_neg_train)

model = create_model(input_shape=(64, 64, 14))

model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(lr=1e-6))

W_old = k_utils.dummy_loader(temp_dir+'{}_pp14_try{}_tune'.format(key, try_))
model.set_weights(W_old)

Y_pred = model.predict([TEST_input_64])
Y_pred[Y_pred<0] = 0
Y_pred[Y_pred>1] = 1
record = verif_metric(TEST_target, Y_pred)

print("Initial record: {}".format(record))

count = 0

for r in range(training_rounds):
    tol = 0 - count - r
    
    model = create_model(input_shape=(64, 64, 14))
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(lr=2e-6))
    
    W_old = k_utils.dummy_loader(temp_dir+'{}_pp14_try{}'.format(key, try_))
    model.set_weights(W_old)
    
    set_seeds(seeds[r])
    print('Training round {}'.format(r))
    
    for i in range(epochs):
    
        #backend.set_value(model.optimizer.learning_rate, learning_rate[i])
        
        #print('epoch = {}'.format(i))
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
                    X_batch_64[k, ..., l] = data[..., c]
                    
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
            model.train_on_batch(X_batch_64, Y_batch);
            
        # epoch end operations
        Y_pred = model.predict([TEST_input_64])
        Y_pred[Y_pred<0] = 0
        Y_pred[Y_pred>1] = 1
        record_temp = verif_metric(TEST_target, Y_pred)

        # if i % 10 == 0:
        #     model.save(model_path_backup)

        if (record - record_temp > min_del):
            print('Validation loss improved from {} to {}'.format(record, record_temp))
            record = record_temp
            tol = 0
            #print('tol: {}'.format(tol))
            # save
            print('save to: {}'.format(model_path))
            model.save(model_path)
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
                    if tol == 1 and i > count:
                        count = i
                        print('count reset: {}'.format(count))
                    continue;
        print("--- %s seconds ---" % (time.time() - start_time))
    
    