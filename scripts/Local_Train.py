# general tools
import os
import sys
import time
import h5py
import random
from glob import glob

import numpy as np
from random import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from keras_unet_collection import utils as k_utils

sys.path.insert(0, '/glade/u/home/ksha/NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/NCAR/libs/')

from namelist import *
import data_utils as du

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('ix', help='ix')
args = vars(parser.parse_args())

# =============== #
IX = int(args['ix'])


def pos_mixer(TRAIN, L, a0=0, a1=0.2):
    data_shape = TRAIN.shape
    out = np.empty((L, data_shape[-1]))
    
    for i in range(L):
        inds = np.random.choice(data_shape[0], 2)
        a = np.random.uniform(a0, a1)
        out[i, :] = a*TRAIN[inds[0], :] + (1-a)*TRAIN[inds[1], :]
    return out

def create_model():

    IN = keras.Input((768,))

    X = IN

    X = keras.layers.Dense(1024, activity_regularizer=keras.regularizers.L2(1e-2))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Dropout(0.3)(X)
    #X = keras.layers.GaussianDropout(0.1)(X)

    X = keras.layers.Dense(512, activity_regularizer=keras.regularizers.L2(1e-2))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Dropout(0.3)(X)
    #X = keras.layers.GaussianDropout(0.1)(X)

    X = keras.layers.Dense(128, activity_regularizer=keras.regularizers.L2(1e-2))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Dropout(0.3)(X)

    X = keras.layers.Dense(64, activity_regularizer=keras.regularizers.L2(1e-2))(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation("gelu")(X)

    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=IN, outputs=OUT)
    
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

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
batch_dir = '/glade/scratch/ksha/DATA/NCAR_batch/'
temp_dir = '/glade/work/ksha/NCAR/Keras_models/'

key = 'HEAD_Lead2'

min_del = 0
max_tol = 10 # early stopping with patience

epochs = 500
batch_size = 200
L_train = 200

with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    land_mask_80km = h5io['land_mask_80km'][...]
    
grid_shape = land_mask_80km.shape

# data_lead1_p0 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead1_part0_vec2.npy', allow_pickle=True)[()]
# data_lead1_p1 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead1_part1_vec2.npy', allow_pickle=True)[()]
# data_lead1_p2 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead1_part2_vec2.npy', allow_pickle=True)[()]

data_lead2_p0 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead2_part0_vec2.npy', allow_pickle=True)[()]
data_lead2_p1 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead2_part1_vec2.npy', allow_pickle=True)[()]
data_lead2_p2 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead2_part2_vec2.npy', allow_pickle=True)[()]

data_lead3_p0 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead3_part0_vec2.npy', allow_pickle=True)[()]
data_lead3_p1 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead3_part1_vec2.npy', allow_pickle=True)[()]
data_lead3_p2 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead3_part2_vec2.npy', allow_pickle=True)[()]

data_lead4_p0 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead4_part0_vec2.npy', allow_pickle=True)[()]
data_lead4_p1 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead4_part1_vec2.npy', allow_pickle=True)[()]
data_lead4_p2 = np.load('/glade/work/ksha/NCAR/TRAIN_pred_lead4_part2_vec2.npy', allow_pickle=True)[()]

TRAIN_256 = np.concatenate((data_lead2_p0['y_vector'], 
                            data_lead2_p1['y_vector'], 
                            data_lead2_p2['y_vector'],
                            data_lead3_p0['y_vector'], 
                            data_lead3_p1['y_vector'], 
                            data_lead3_p2['y_vector'],
                            data_lead4_p0['y_vector'], 
                            data_lead4_p1['y_vector'], 
                            data_lead4_p2['y_vector'],), axis=0)


TRAIN_pred = np.concatenate((data_lead2_p0['y_pred'], 
                             data_lead2_p1['y_pred'], 
                             data_lead2_p2['y_pred'],
                             data_lead3_p0['y_pred'], 
                             data_lead3_p1['y_pred'], 
                             data_lead3_p2['y_pred'],
                             data_lead4_p0['y_pred'], 
                             data_lead4_p1['y_pred'], 
                             data_lead4_p2['y_pred'],), axis=0)

TRAIN_Y = np.concatenate((data_lead2_p0['y_true'], 
                          data_lead2_p1['y_true'], 
                          data_lead2_p2['y_true'],
                          data_lead3_p0['y_true'], 
                          data_lead3_p1['y_true'], 
                          data_lead3_p2['y_true'],
                          data_lead4_p0['y_true'], 
                          data_lead4_p1['y_true'], 
                          data_lead4_p2['y_true'],), axis=0)

TRAIN_256_pick = TRAIN_256 #[flag_pick_train, :]
TRAIN_pred_pick = TRAIN_pred #[flag_pick_train, :]
TRAIN_Y_pick = TRAIN_Y #[flag_pick_train]

TRAIN_256_pos = TRAIN_256[TRAIN_Y==1]
TRAIN_256_neg = TRAIN_256[TRAIN_Y==0]

filename_valid = sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch/VALID*neg_neg_neg*lead2.npy")) + \
                 sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch/VALID*pos*lead2.npy"))

data_p_valid = np.load('/glade/work/ksha/NCAR/TEST_pred_lead2_vec2.npy', allow_pickle=True)[()]

filename_test = filename_valid + \
                sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v4/*neg_neg_neg*lead{}.npy".format(2))) + \
                sorted(glob("/glade/scratch/ksha/DATA/NCAR_batch_v4/*pos*lead{}.npy".format(2)))

data_p_test = np.load('/glade/work/ksha/NCAR/TEST_pred_lead2_v4_vec2.npy', allow_pickle=True)[()]

TEST_256 = np.concatenate((data_p_valid['y_vector'], data_p_test['y_vector']), axis=0)
TEST_pred = np.concatenate((data_p_valid['y_pred'], data_p_test['y_pred']), axis=0)
TEST_Y = np.concatenate((data_p_valid['y_true'], data_p_test['y_true']), axis=0)

ix = IX
for iy in range(grid_shape[1]-1, -1, -1):
    if land_mask_80km[ix, iy]:

        name_block5 = []

        for i in range(ix-2, ix+3):
            for j in range(iy-2, iy+3):
                name_block5.append('indx{}_indy{}'.format(i, j))

        L_test = len(filename_test)

        flag_pick_test = [False,]*L_test
        filename_pick_test = []

        for i, name in enumerate(filename_test):
            for patterns in name_block5:
                if patterns in name:
                    flag_pick_test[i] = True
                    filename_pick_test.append(name)
                    break;

        # ========== Sample Checks ========== #

        N_samples = np.sum(np.array(flag_pick_test))

        if N_samples < 1:
            continue;

        TEST_256_pick = TEST_256[flag_pick_test, :]
        TEST_pred_pick = TEST_pred[flag_pick_test, :]
        TEST_Y_pick = TEST_Y[flag_pick_test]

        if np.sum(TEST_Y_pick) < 25:
            continue;

        ref = np.sum(TEST_Y_pick) / len(TEST_Y_pick)

        # =========== Model Section ========== #

        model_name = '{}_ix{}_iy{}'.format(key, ix, iy)
        model_path = temp_dir+model_name

        model = create_model()

        model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
                      optimizer=keras.optimizers.Adam(lr=1e-5))

        tol = 0

        print('===== ix:{}; iy:{} starts ====='.format(ix, iy))

        # ========== Training loop ========== #
        L_pos = len(TRAIN_256_pos)
        L_neg = len(TRAIN_256_neg)

        training_rounds = 10
        seeds = [12342, 2536234, 98765, 473, 865, 7456, 69472, 3456357, 3425, 678]

        record = 1.1
        print("Initial record: {}".format(record))


        min_del = 0
        max_tol = 10 # early stopping with patience

        epochs = 500
        batch_size = 200
        L_train = 200 #int(len(TRAIN_Y_pick) / batch_size)

        for r in range(training_rounds):
            if r == 0:
                tol = 0
            else:
                tol = -200

            model = create_model()

            model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
                          optimizer=keras.optimizers.Adam(lr=1e-5))

            set_seeds(seeds[r])
            print('Training round {}'.format(r))

            for i in range(epochs):

                #backend.set_value(model.optimizer.learning_rate, learning_rate[i])

                #print('epoch = {}'.format(i))
                start_time = time.time()

                # loop of batch
                for j in range(L_train):

                    #N_aug = int(np.random.uniform(2, 7))
                    #N_pos = int(np.random.uniform(30, 50))
                    N_pos = 100
                    #N_aug = int(np.random.uniform(20, 45))

                    N_neg = batch_size - N_pos

                    ind_neg = du.shuffle_ind(L_neg)
                    ind_pos = du.shuffle_ind(L_pos)

                    ind_neg_pick = ind_neg[:N_neg]
                    ind_pos_pick = ind_pos[:N_pos]

                    X_batch_neg = TRAIN_256_neg[ind_neg_pick, :] #
                    X_batch_pos = pos_mixer(TRAIN_256_pos, N_pos, a0=0, a1=0.05) # 

                    # np.random.shuffle(TRAIN_256_neg)
                    # np.random.shuffle(TRAIN_256_pos)

                    # X_batch_neg = TRAIN_256_neg[:N_neg, :]
                    # X_batch_pos = TRAIN_256_pos[:N_pos, :]

                    X_batch = np.concatenate((X_batch_neg, X_batch_pos), axis=0)

                    Y_batch = np.concatenate((np.random.uniform(0.0, 0.01, size=N_neg), 
                                              np.random.uniform(0.98, 0.999, size=N_pos)), axis=0)

                    # Y_batch = np.ones([batch_size,])
                    # Y_batch[:N_neg] = 0.0

                    ind_ = du.shuffle_ind(batch_size)

                    X_batch = X_batch[ind_, :]
                    Y_batch = Y_batch[ind_]

                    # train on batch
                    model.train_on_batch(X_batch, Y_batch);

                # epoch end operations
                Y_pred = model.predict([TEST_256_pick])

                Y_pred[Y_pred<0] = 0
                Y_pred[Y_pred>1] = 1

                record_temp = verif_metric(TEST_Y_pick, Y_pred, ref)

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
                    if record_temp > 1.0:
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


    