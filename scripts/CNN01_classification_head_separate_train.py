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
import graph_utils as gu

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

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

def verif_metric(VALID_target, Y_pred, ref):


    # fpr, tpr, thresholds = roc_curve(VALID_target.ravel(), Y_pred.ravel())
    # AUC = auc(fpr, tpr)
    # AUC_metric = 1 - AUC
    
    BS = np.mean((VALID_target.ravel() - Y_pred.ravel())**2)
    #ll = log_loss(VALID_target.ravel(), Y_pred.ravel())
    
    #print('{}'.format(BS))
    metric = BS

    return metric / ref

lead1 = 2
lead2 = 3
lead3 = 4
lead4 = 5
lead5 = 6
lead6 = 20
lead7 = 21
lead8 = 22
lead9 = 23

lead_name = 4
model_tag = 're'

filepath_vec = "/glade/work/ksha/NCAR/"

# Training set (feature vectors)

data_lead1_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead1_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead1_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]

data_lead2_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead2_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead2_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]

data_lead3_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead3_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead3_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]

data_lead4_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]
data_lead4_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]
data_lead4_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]

data_lead5_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead5, model_tag), allow_pickle=True)[()]
data_lead5_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead5, model_tag), allow_pickle=True)[()]
data_lead5_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead5, model_tag), allow_pickle=True)[()]

data_lead6_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead6, model_tag), allow_pickle=True)[()]
data_lead6_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead6, model_tag), allow_pickle=True)[()]
data_lead6_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead6, model_tag), allow_pickle=True)[()]

data_lead7_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead7, model_tag), allow_pickle=True)[()]
data_lead7_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead7, model_tag), allow_pickle=True)[()]
data_lead7_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead7, model_tag), allow_pickle=True)[()]

data_lead8_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead8, model_tag), allow_pickle=True)[()]
data_lead8_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead8, model_tag), allow_pickle=True)[()]
data_lead8_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead8, model_tag), allow_pickle=True)[()]

data_lead9_p0 = np.load('{}TRAIN_v3_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead9, model_tag), allow_pickle=True)[()]
data_lead9_p1 = np.load('{}TRAIN_v3_vec_lead{}_part1_{}.npy'.format(filepath_vec, lead9, model_tag), allow_pickle=True)[()]
data_lead9_p2 = np.load('{}TRAIN_v3_vec_lead{}_part2_{}.npy'.format(filepath_vec, lead9, model_tag), allow_pickle=True)[()]

TRAIN_lead1_v3 = np.concatenate((data_lead1_p0['y_vector'], data_lead1_p1['y_vector'], data_lead1_p2['y_vector']), axis=0)
TRAIN_lead2_v3 = np.concatenate((data_lead2_p0['y_vector'], data_lead2_p1['y_vector'], data_lead2_p2['y_vector']), axis=0)
TRAIN_lead3_v3 = np.concatenate((data_lead3_p0['y_vector'], data_lead3_p1['y_vector'], data_lead3_p2['y_vector']), axis=0)
TRAIN_lead4_v3 = np.concatenate((data_lead4_p0['y_vector'], data_lead4_p1['y_vector'], data_lead4_p2['y_vector']), axis=0)
TRAIN_lead5_v3 = np.concatenate((data_lead5_p0['y_vector'], data_lead5_p1['y_vector'], data_lead5_p2['y_vector']), axis=0)
TRAIN_lead6_v3 = np.concatenate((data_lead6_p0['y_vector'], data_lead6_p1['y_vector'], data_lead6_p2['y_vector']), axis=0)
TRAIN_lead7_v3 = np.concatenate((data_lead7_p0['y_vector'], data_lead7_p1['y_vector'], data_lead7_p2['y_vector']), axis=0)
TRAIN_lead8_v3 = np.concatenate((data_lead8_p0['y_vector'], data_lead8_p1['y_vector'], data_lead8_p2['y_vector']), axis=0)
TRAIN_lead9_v3 = np.concatenate((data_lead9_p0['y_vector'], data_lead9_p1['y_vector'], data_lead9_p2['y_vector']), axis=0)

TRAIN_lead1_y_v3 = np.concatenate((data_lead1_p0['y_true'], data_lead1_p1['y_true'], data_lead1_p2['y_true']), axis=0)
TRAIN_lead2_y_v3 = np.concatenate((data_lead2_p0['y_true'], data_lead2_p1['y_true'], data_lead2_p2['y_true']), axis=0)
TRAIN_lead3_y_v3 = np.concatenate((data_lead3_p0['y_true'], data_lead3_p1['y_true'], data_lead3_p2['y_true']), axis=0)
TRAIN_lead4_y_v3 = np.concatenate((data_lead4_p0['y_true'], data_lead4_p1['y_true'], data_lead4_p2['y_true']), axis=0)
TRAIN_lead5_y_v3 = np.concatenate((data_lead5_p0['y_true'], data_lead5_p1['y_true'], data_lead5_p2['y_true']), axis=0)
TRAIN_lead6_y_v3 = np.concatenate((data_lead6_p0['y_true'], data_lead6_p1['y_true'], data_lead6_p2['y_true']), axis=0)
TRAIN_lead7_y_v3 = np.concatenate((data_lead7_p0['y_true'], data_lead7_p1['y_true'], data_lead7_p2['y_true']), axis=0)
TRAIN_lead8_y_v3 = np.concatenate((data_lead8_p0['y_true'], data_lead8_p1['y_true'], data_lead8_p2['y_true']), axis=0)
TRAIN_lead9_y_v3 = np.concatenate((data_lead9_p0['y_true'], data_lead9_p1['y_true'], data_lead9_p2['y_true']), axis=0)

data_lead1_px = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
data_lead2_px = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
data_lead3_px = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
data_lead4_px = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]
data_lead5_px = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead5, model_tag), allow_pickle=True)[()]
data_lead6_px = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead6, model_tag), allow_pickle=True)[()]
data_lead7_px = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead7, model_tag), allow_pickle=True)[()]
data_lead8_px = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead8, model_tag), allow_pickle=True)[()]
data_lead9_px = np.load('{}TRAIN_v4x_vec_lead{}_part0_{}.npy'.format(filepath_vec, lead9, model_tag), allow_pickle=True)[()]

TRAIN_X = np.concatenate((TRAIN_lead1_v3, TRAIN_lead2_v3, TRAIN_lead3_v3, 
                          TRAIN_lead4_v3, TRAIN_lead5_v3, TRAIN_lead6_v3, 
                          TRAIN_lead7_v3, TRAIN_lead8_v3, TRAIN_lead9_v3,
                          data_lead1_px['y_vector'], data_lead2_px['y_vector'],
                          data_lead3_px['y_vector'], data_lead4_px['y_vector'],
                          data_lead5_px['y_vector'], data_lead6_px['y_vector'],
                          data_lead7_px['y_vector'], data_lead8_px['y_vector'],
                          data_lead9_px['y_vector']), axis=0)


TRAIN_Y = np.concatenate((TRAIN_lead1_y_v3, TRAIN_lead2_y_v3, TRAIN_lead3_y_v3, 
                          TRAIN_lead4_y_v3, TRAIN_lead5_y_v3, TRAIN_lead6_y_v3, 
                          TRAIN_lead7_y_v3, TRAIN_lead8_y_v3, TRAIN_lead9_y_v3,
                          data_lead1_px['y_true'], data_lead2_px['y_true'],
                          data_lead3_px['y_true'], data_lead4_px['y_true'],
                          data_lead5_px['y_true'], data_lead6_px['y_true'],
                          data_lead7_px['y_true'], data_lead8_px['y_true'],
                          data_lead9_px['y_true']), axis=0)

TRAIN_X_pos = TRAIN_X[TRAIN_Y == 1]
TRAIN_X_neg = TRAIN_X[TRAIN_Y == 0]

# Validation set

valid_lead1 = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
valid_lead2 = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
valid_lead3 = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
valid_lead4 = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]
valid_lead5 = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead5, model_tag), allow_pickle=True)[()]
valid_lead6 = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead6, model_tag), allow_pickle=True)[()]
valid_lead7 = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead7, model_tag), allow_pickle=True)[()]
valid_lead8 = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead8, model_tag), allow_pickle=True)[()]
valid_lead9 = np.load('{}VALID_v3_vec_lead{}_{}.npy'.format(filepath_vec, lead9, model_tag), allow_pickle=True)[()]

valid_lead1_px = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead1, model_tag), allow_pickle=True)[()]
valid_lead2_px = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead2, model_tag), allow_pickle=True)[()]
valid_lead3_px = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead3, model_tag), allow_pickle=True)[()]
valid_lead4_px = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead4, model_tag), allow_pickle=True)[()]
valid_lead5_px = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead5, model_tag), allow_pickle=True)[()]
valid_lead6_px = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead6, model_tag), allow_pickle=True)[()]
valid_lead7_px = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead7, model_tag), allow_pickle=True)[()]
valid_lead8_px = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead8, model_tag), allow_pickle=True)[()]
valid_lead9_px = np.load('{}VALID_v4x_vec_lead{}_{}.npy'.format(filepath_vec, lead9, model_tag), allow_pickle=True)[()]

VALID_X = np.concatenate((valid_lead1['y_vector'], valid_lead2['y_vector'],
                          valid_lead3['y_vector'], valid_lead4['y_vector'],
                          valid_lead5['y_vector'], valid_lead6['y_vector'],
                          valid_lead7['y_vector'], valid_lead8['y_vector'],
                          valid_lead9['y_vector'], valid_lead1_px['y_vector'], 
                          valid_lead2_px['y_vector'], valid_lead3_px['y_vector'], 
                          valid_lead4_px['y_vector'], valid_lead5_px['y_vector'], 
                          valid_lead6_px['y_vector'], valid_lead7_px['y_vector'],
                          valid_lead8_px['y_vector'], valid_lead9_px['y_vector']), axis=0)

VALID_Y = np.concatenate((valid_lead1['y_true'], valid_lead2['y_true'],
                          valid_lead3['y_true'], valid_lead4['y_true'],
                          valid_lead5['y_true'], valid_lead6['y_true'],
                          valid_lead7['y_true'], valid_lead8['y_true'],
                          valid_lead9['y_true'], valid_lead1_px['y_true'], 
                          valid_lead2_px['y_true'], valid_lead3_px['y_true'], 
                          valid_lead4_px['y_true'], valid_lead5_px['y_true'], 
                          valid_lead6_px['y_true'], valid_lead7_px['y_true'],
                          valid_lead8_px['y_true'], valid_lead9_px['y_true']), axis=0)

seeds = [12342, 2536234, 98765, 473, 865, 7456, 69472, 3456357, 3425, 678,
         2452624, 5787, 235362, 67896, 98454, 12445, 46767, 78906, 345, 8695, 
         2463725, 4734, 23234, 884, 2341, 362, 5, 234, 483, 785356, 23425, 3621, 
         58461, 80968765, 123, 425633, 5646, 67635, 76785, 34214]

training_rounds = len(seeds)

ref = np.sum(VALID_Y) / len(VALID_Y)

# =========== Model Section ========== #

batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch/'
temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

key = 'HEAD_base3'

model_name = '{}'.format(key)
model_path = temp_dir+model_name

tol = 0

# ========== Training loop ========== #
L_pos = len(TRAIN_X_pos)
L_neg = len(TRAIN_X_neg)

record = 1.1 #record_start
print("Initial record: {}".format(record))

min_del = 0
max_tol = 100 # early stopping with patience

epochs = 500
batch_size = 64
L_train = 16

for r in range(training_rounds):
    if r == 0:
        tol = 0
    else:
        tol = -200

    model_head = create_model_head()
    model_head.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=keras.optimizers.Adam(lr=1e-4))
    
    set_seeds(int(seeds[r]))
    print('Training round {}'.format(r))

    for i in range(epochs):            
        start_time = time.time()

        # loop of batch
        for j in range(L_train):
            N_pos = 32
            N_neg = batch_size - N_pos

            ind_neg = du.shuffle_ind(L_neg)
            ind_pos = du.shuffle_ind(L_pos)

            ind_neg_pick = ind_neg[:N_neg]
            ind_pos_pick = ind_pos[:N_pos]

            X_batch_neg = TRAIN_X_neg[ind_neg_pick, :]
            X_batch_pos = TRAIN_X_pos[ind_pos_pick, :]
            X_batch = np.concatenate((X_batch_neg, X_batch_pos), axis=0)
            
            Y_batch = np.ones([batch_size,])
            Y_batch[:N_neg] = 0.0

            ind_ = du.shuffle_ind(batch_size)

            X_batch = X_batch[ind_, :]
            Y_batch = Y_batch[ind_]

            # train on batch
            model_head.train_on_batch(X_batch, Y_batch);
            
        if i > 8:
            # epoch end operations
            Y_pred = model_head.predict(VALID_X)

            Y_pred[Y_pred<0] = 0
            Y_pred[Y_pred>1] = 1

            record_temp = verif_metric(VALID_Y, Y_pred, ref)

            # if i % 10 == 0:
            #     model.save(model_path_backup)

            if (record - record_temp >= min_del):
                print('Validation loss improved from {} to {}'.format(record, record_temp))
                record = record_temp
                tol = 0

                #print('tol: {}'.format(tol))
                # save
                print('save to: {}'.format(model_path))
                model_head.save(model_path)
            else:
                print('Validation loss {} NOT improved'.format(record_temp))
                if record_temp > 1.1:
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

