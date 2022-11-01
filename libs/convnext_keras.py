
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import Sequential
from tensorflow.keras import Model

class StochasticDepth(layers.Layer):
    """Stochastic Depth module.
    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.
    References:
      - https://github.com/rwightman/pytorch-image-models
    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].
    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * 3#(len(tf.shape(x)) - 1) #
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config
    
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
    
def ConvNeXtBlock(projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, name=None):

    if name is None:
        name = "prestem" + str(backend.get_uid("prestem"))

    def apply(inputs):
        x = inputs

        x = layers.Conv2D(filters=projection_dim, kernel_size=7, padding="same",
            groups=projection_dim, name=name + "_depthwise_conv")(x)
        
        x = layers.LayerNormalization(epsilon=1e-6, name=name + "_layernorm")(x)
        x = layers.Dense(4 * projection_dim, name=name + "_pointwise_conv_1")(x)
        x = layers.Activation("gelu", name=name + "_gelu")(x)
        x = layers.Dense(projection_dim, name=name + "_pointwise_conv_2")(x)

        if layer_scale_init_value is not None:
            x = LayerScale(layer_scale_init_value, projection_dim,
                           name=name + "_layer_scale")(x)
        if drop_path_rate:
            layer = StochasticDepth(drop_path_rate, name=name + "_stochastic_depth")
        else:
            layer = layers.Activation("linear", name=name + "_identity")

        return inputs + layer(x)

    return apply

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

def ConvNeXt(
    depths,
    projection_dims,
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    model_name="convnext",
    input_shape=None,
    pooling=None,
    classes=1000,
):

    inputs = layers.Input(shape=input_shape)
    x = inputs


    # Stem block.
    stem = Sequential(
        [
            layers.Conv2D(
                projection_dims[0],
                kernel_size=4,
                strides=4,
                name=model_name + "_stem_conv",
            ),
            layers.LayerNormalization(
                epsilon=1e-6, name=model_name + "_stem_layernorm"
            ),
        ],
        name=model_name + "_stem",
    )

    # Downsampling blocks.
    downsample_layers = []
    downsample_layers.append(stem)

    num_downsample_layers = 3
    for i in range(num_downsample_layers):
        downsample_layer = Sequential(
            [
                layers.LayerNormalization(
                    epsilon=1e-6,
                    name=model_name + "_downsampling_layernorm_" + str(i),
                ),
                layers.Conv2D(
                    projection_dims[i + 1],
                    kernel_size=2,
                    strides=2,
                    name=model_name + "_downsampling_conv_" + str(i),
                ),
            ],
            name=model_name + "_downsampling_block_" + str(i),
        )
        downsample_layers.append(downsample_layer)

    depth_drop_rates = [
        float(x) for x in np.linspace(0.0, drop_path_rate, sum(depths))
    ]

    # First apply downsampling blocks and then apply ConvNeXt stages.
    cur = 0

    num_convnext_blocks = 4
    for i in range(num_convnext_blocks):
        x = downsample_layers[i](x)
        for j in range(depths[i]):
            x = ConvNeXtBlock(
                projection_dim=projection_dims[i],
                drop_path_rate=depth_drop_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value,
                name=model_name + f"_stage_{i}_block_{j}",
            )(x)
        cur += depths[i]

    x = Head(num_classes=classes, name=model_name)(x)
    
    model = Model(inputs=inputs, outputs=x, name=model_name)

    return model

