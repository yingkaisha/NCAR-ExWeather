import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

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
    

def dummy_loader(model_path):
    '''
    Load a stored keras model and return its weights.
    
    Input
    ----------
        The file path of the stored keras model.
    
    Output
    ----------
        Weights of the model.
        
    '''
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W
    
def create_model_base(input_shape=(64, 64, 15), depths=[3, 3, 27, 3], projection_dims=[32, 32, 64, 64], first_pool=4):

    
    drop_path_rate=0.0
    layer_scale_init_value=1e-6

    model_name='Branch64X'
    IN64 = keras.layers.Input(shape=input_shape)
    X = IN64
    # ----- convnext block 0 ----- #

    X = keras.layers.Conv2D(projection_dims[0], kernel_size=first_pool, strides=first_pool, name="{}_down0".format(model_name))(X)
    X = keras.layers.LayerNormalization(epsilon=1e-6, name="{}_down0_norm".format(model_name))(X)

    for j in range(depths[0]):

        X_convnext = X
        X_convnext = keras.layers.Conv2D(filters=projection_dims[0], kernel_size=7, padding="same",
                              groups=projection_dims[0], name="{}_down0_dconv{}".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.LayerNormalization(epsilon=1e-6, name="{}_down0_dconv{}_norm".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Dense(4 * projection_dims[0], name="{}_down0_dense{}_p1".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Activation("gelu", name="{}_down0_gelu{}".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Dense(projection_dims[0], name="{}_down0_dense{}_p2".format(model_name, j))(X_convnext)

        X_convnext = LayerScale(layer_scale_init_value, projection_dims[0], name="{}_down0_layerscale{}".format(model_name, j))(X_convnext)

        X = X + X_convnext


    # ----- convnext block 1 ----- #

    X = keras.layers.LayerNormalization(epsilon=1e-6, name="{}_down1_norm".format(model_name))(X)
    X = keras.layers.Conv2D(projection_dims[1], kernel_size=2, strides=2, name="{}_down1".format(model_name))(X)

    for j in range(depths[1]):

        X_convnext = X
        X_convnext = keras.layers.Conv2D(filters=projection_dims[1], kernel_size=7, padding="same",
                                   groups=projection_dims[1], name="{}_down1_dconv{}".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.LayerNormalization(epsilon=1e-6, name="{}_down1_dconv{}_norm".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Dense(4 * projection_dims[1], name="{}_down1_dense{}_p1".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Activation("gelu", name="{}_down1_gelu{}".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Dense(projection_dims[1], name="{}_down1_dense{}_p2".format(model_name, j))(X_convnext)

        X_convnext = LayerScale(layer_scale_init_value, projection_dims[1], name="{}_down1_layerscale{}".format(model_name, j))(X_convnext)

        X = X + X_convnext

    # ----- convnext block 2 ----- #

    X = keras.layers.LayerNormalization(epsilon=1e-6, name="{}_down2_norm".format(model_name))(X)
    X = keras.layers.Conv2D(projection_dims[2], kernel_size=2, strides=2, name="{}_down2".format(model_name))(X)

    for j in range(depths[2]):

        X_convnext = X
        X_convnext = keras.layers.Conv2D(filters=projection_dims[2], kernel_size=5, padding="same",
                                   groups=projection_dims[2], name="{}_down2_dconv{}".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.LayerNormalization(epsilon=1e-6, name="{}_down2_dconv{}_norm".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Dense(4 * projection_dims[2], name="{}_down2_dense{}_p1".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Activation("gelu", name="{}_down2_gelu{}".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Dense(projection_dims[2], name="{}_down2_dense{}_p2".format(model_name, j))(X_convnext)

        X_convnext = LayerScale(layer_scale_init_value, projection_dims[2], name="{}_down2_layerscale{}".format(model_name, j))(X_convnext)

        X = X + X_convnext

    # ----- convnext block 3 ----- #

    X = keras.layers.LayerNormalization(epsilon=1e-6, name="{}_down3_norm".format(model_name))(X)
    X = keras.layers.Conv2D(projection_dims[3], kernel_size=2, padding='same', name="{}_down3".format(model_name))(X)

    for j in range(depths[3]):

        X_convnext = X
        X_convnext = keras.layers.Conv2D(filters=projection_dims[3], kernel_size=5, padding="same",
                                   groups=projection_dims[3], name="{}_down3_dconv{}".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.LayerNormalization(epsilon=1e-6, name="{}_down3_dconv{}_norm".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Dense(4 * projection_dims[3], name="{}_down3_dense{}_p1".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Activation("gelu", name="{}_down3_gelu{}".format(model_name, j))(X_convnext)
        X_convnext = keras.layers.Dense(projection_dims[3], name="{}_down3_dense{}_p2".format(model_name, j))(X_convnext)

        X_convnext = LayerScale(layer_scale_init_value, projection_dims[3], name="{}_down3_layerscale{}".format(model_name, j))(X_convnext)

        X = X + X_convnext

    V1 = X

    OUT = keras.layers.GlobalMaxPooling2D(name="{}_head_pool64".format(model_name))(V1)
    model = keras.Model(inputs=IN64, outputs=OUT, name=model_name)
    
    return model


def create_model_vgg(input_shape=(32, 32, 15), channels = [32, 64, 96, 128]):
    
    IN = layers.Input(shape=input_shape)

    X = IN

    X = keras.layers.Conv2D(channels[0], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[0], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)
    # pooling
    X = keras.layers.Conv2D(channels[1], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

    X = keras.layers.Conv2D(channels[1], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[1], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    # pooling
    X = keras.layers.Conv2D(channels[2], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

    X = keras.layers.Conv2D(channels[2], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[2], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    # pooling
    X = keras.layers.Conv2D(channels[3], kernel_size=2, strides=(2, 2), padding='valid', use_bias=True)(X)

    X = keras.layers.Conv2D(channels[3], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    X = keras.layers.Conv2D(channels[3], kernel_size=3, padding='same', use_bias=False)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation("gelu")(X)

    V1 = X
    OUT = keras.layers.GlobalMaxPooling2D()(V1)
    model = Model(inputs=IN, outputs=OUT)
    
    return model

def create_model_head(input_shape=(128,), N_node=64):
    
    IN_vec = keras.Input(input_shape)    
    X = IN_vec
    #
    X = keras.layers.Dense(N_node)(X)
    X = keras.layers.Activation("relu")(X)
    X = keras.layers.BatchNormalization()(X)
    
    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=IN_vec, outputs=OUT)
    
    return model

def create_classif_head():
    
    IN = keras.Input((L_vec, 128))
    X = IN
    X = keras.layers.Conv1D(128, kernel_size=2, strides=1, padding='valid')(X)
    X = keras.layers.Activation("gelu")(X)
    
    #
    IN_vec = keras.Input((2,))
    
    X = keras.layers.GlobalMaxPool1D()(X) #X = keras.layers.Flatten()(X)
    X = keras.layers.Concatenate()([X, IN_vec])
    
    X = keras.layers.Dense(64)(X)
    X = keras.layers.Activation("relu")(X)
    X = keras.layers.BatchNormalization()(X)

    OUT = X
    OUT = keras.layers.Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(-10))(OUT)

    model = keras.models.Model(inputs=[IN, IN_vec], outputs=OUT)
    return model

def verif_metric(VALID_target, Y_pred):
    
    BS = np.mean((VALID_target.ravel() - Y_pred.ravel())**2)
    
    print('{}'.format(BS))
    metric = BS

    return metric
    
def name_extract(filenames):
    '''Separate train, valid, and test patches based on their filenames.
      Returns 2 lists of filenames: filename_train, filename_valid'''
    
    date_train_end = datetime(2020, 7, 14)
    date_valid_end = datetime(2021, 1, 1)
    
    filename_train = []
    filename_valid = []
    
    # --------------------------------------- #
    base_v3_s = datetime(2018, 7, 15)
    base_v4_s = datetime(2020, 12, 3)
    base_v4x_s = datetime(2019, 10, 1)
    
    date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
    date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+365+30)]
    date_list_v4x = [base_v4x_s + timedelta(days=day) for day in range(429)]
    # ---------------------------------------- #
    
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
        
        if (day - date_train_end).days < 0:
            filename_train.append(name)
            
        else:
            if (day - date_valid_end).days < 0:
                filename_valid.append(name)

    return filename_train, filename_valid
    
    
def feature_extract(filenames, lon_80km, lon_minmax, lat_80km, lat_minmax, elev_80km, elev_max):
    
    lon_out = []
    lat_out = []
    elev_out = []
    mon_out = []
    
    base_v3_s = datetime(2018, 7, 15)
    base_v3_e = datetime(2020, 12, 2)

    base_v4_s = datetime(2020, 12, 3)
    base_v4_e = datetime(2022, 7, 15)
    
    date_list_v3 = [base_v3_s + timedelta(days=day) for day in range(365+365+142)]
    date_list_v4 = [base_v4_s + timedelta(days=day) for day in range(365+180-151)]
    
    for i, name in enumerate(filenames):
        
        if 'v4' in name:
            date_list = date_list_v4
        else:
            date_list = date_list_v3
        
        nums = re.findall(r'\d+', name)
        indy = int(nums[-2])
        indx = int(nums[-3])
        day = int(nums[-4])
        day = date_list[day]
        month = day.month
        
        month_norm = (month - 1)/(12-1)
        
        lon = lon_80km[indx, indy]
        lat = lat_80km[indx, indy]

        lon = (lon - lon_minmax[0])/(lon_minmax[1] - lon_minmax[0])
        lat = (lat - lat_minmax[0])/(lat_minmax[1] - lat_minmax[0])

        elev = elev_80km[indx, indy]
        elev = elev / elev_max
        
        lon_out.append(lon)
        lat_out.append(lat)
        elev_out.append(elev)
        mon_out.append(month_norm)
        
    return np.array(lon_out), np.array(lat_out), np.array(elev_out), np.array(mon_out)
    
    
def name_to_ind(filenames):
    
    indx_out = []
    indy_out = []
    day_out = []
    flag_out = []
    
    for i, name in enumerate(filenames):
        nums = re.findall(r'\d+', name)
        indy = int(nums[-2])
        indx = int(nums[-3])
        day = int(nums[-4])
        
        indx_out.append(indx)
        indy_out.append(indy)
        day_out.append(day)
        
        if "pos" in name:
            flag_out.append(True)
        else:
            flag_out.append(False)
        
    return np.array(indx_out), np.array(indy_out), np.array(day_out), np.array(flag_out)

    
    
    
    