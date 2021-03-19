from functools import reduce

# from keras import layers
# from keras import initializers
# from keras import models
# from keras_ import EfficientNetB0, EfficientNetB1, EfficientNetB2
# from keras_ import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as KL
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization
from layers import bbox_transform_inv
from initializers import PriorProbability
from utils.anchors import anchors_for_shape, anchor_targets_bbox, AnchorParameters
import numpy as np
from utils.graph_funcs import trim_zeros_graph, overlaps_graph, batch_slice
from losses import focal, smooth_l1

w_bifpns = [64, 88, 112, 160, 224, 288, 384]
d_bifpns = [3, 4, 5, 6, 7, 7, 8]
d_heads = [3, 3, 3, 4, 4, 4, 5]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]

MOMENTUM = 0.997
EPSILON = 1e-4
AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    # ratio=h/w
    ratios=np.array([1, 0.5, 2], keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)
ANCHOR_PARAMETERS = AnchorParameters.default
MAX_INSTANCES = 100
TRAIN_ROIS_PER_IMAGE = 200

def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = KL.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=f'{name}/conv')
    f2 = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def ConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = KL.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                       use_bias=True, name='{}_conv'.format(name))
    f2 = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='{}_bn'.format(name))
    # f2 = BatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = KL.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def build_wBiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = KL.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        # P6_in = BatchNormalization(freeze=freeze_bn, name='resample_p6/bn')(P6_in)
        P6_in = KL.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        P7_in = KL.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P7_U = KL.UpSampling2D()(P7_in)
        P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = KL.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = KL.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        # P5_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P6_U = KL.UpSampling2D()(P6_td)
        P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
        P5_td = KL.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = KL.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        # P4_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P5_U = KL.UpSampling2D()(P5_td)
        P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
        P4_td = KL.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P3_in = KL.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        # P3_in = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        P4_U = KL.UpSampling2D()(P4_td)
        P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = KL.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P4_in_2 = KL.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        P3_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = KL.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P5_in_2 = KL.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        P4_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
        P5_out = KL.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = KL.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = KL.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P7_U = KL.UpSampling2D()(P7_in)
        P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = KL.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P6_U = KL.UpSampling2D()(P6_td)
        P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
        P5_td = KL.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P5_U = KL.UpSampling2D()(P5_td)
        P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
        P4_td = KL.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P4_U = KL.UpSampling2D()(P4_td)
        P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = KL.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P3_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
        P4_out = KL.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P4_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
        P5_out = KL.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = KL.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = KL.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)
    return P3_out, P4_td, P5_td, P6_td, P7_out


def build_BiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = KL.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name='resample_p6/bn')(P6_in)
        # P6_in = BatchNormalization(freeze=freeze_bn, name='resample_p6/bn')(P6_in)
        P6_in = KL.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        P7_in = KL.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P7_U = KL.UpSampling2D()(P7_in)
        P6_td = KL.Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = KL.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P5_in_1 = KL.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        # P5_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P6_U = KL.UpSampling2D()(P6_td)
        P5_td = KL.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in_1, P6_U])
        P5_td = KL.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P4_in_1 = KL.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        # P4_in_1 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P5_U = KL.UpSampling2D()(P5_td)
        P4_td = KL.Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in_1, P5_U])
        P4_td = KL.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P3_in = KL.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        # P3_in = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        P4_U = KL.UpSampling2D()(P4_td)
        P3_out = KL.Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = KL.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P4_in_2 = KL.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        # P4_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        P3_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = KL.Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in_2, P4_td, P3_D])
        P4_out = KL.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P5_in_2 = KL.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        P4_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = KL.Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in_2, P5_td, P4_D])
        P5_out = KL.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = KL.Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = KL.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = KL.Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = KL.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        P7_U = KL.UpSampling2D()(P7_in)
        P6_td = KL.Add(name=f'fpn_cells/cell_{id}/fnode0/add')([P6_in, P7_U])
        P6_td = KL.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        P6_U = KL.UpSampling2D()(P6_td)
        P5_td = KL.Add(name=f'fpn_cells/cell_{id}/fnode1/add')([P5_in, P6_U])
        P5_td = KL.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        P5_U = KL.UpSampling2D()(P5_td)
        P4_td = KL.Add(name=f'fpn_cells/cell_{id}/fnode2/add')([P4_in, P5_U])
        P4_td = KL.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        P4_U = KL.UpSampling2D()(P4_td)
        P3_out = KL.Add(name=f'fpn_cells/cell_{id}/fnode3/add')([P3_in, P4_U])
        P3_out = KL.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        P3_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = KL.Add(name=f'fpn_cells/cell_{id}/fnode4/add')([P4_in, P4_td, P3_D])
        P4_out = KL.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        P4_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = KL.Add(name=f'fpn_cells/cell_{id}/fnode5/add')([P5_in, P5_td, P4_D])
        P5_out = KL.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        P5_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = KL.Add(name=f'fpn_cells/cell_{id}/fnode6/add')([P6_in, P6_td, P5_D])
        P6_out = KL.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        P6_D = KL.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = KL.Add(name=f'fpn_cells/cell_{id}/fnode7/add')([P7_in, P6_D])
        P7_out = KL.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)
    return P3_out, P4_td, P5_td, P6_td, P7_out


class BoxNet(KL.Layer):
    def __init__(self, width, depth, num_anchors=9, separable_conv=True, freeze_bn=False, detect_quadrangle=False, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        self.detect_quadrangle = detect_quadrangle
        num_values = 9 if detect_quadrangle else 4
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }
        if separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [KL.SeparableConv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in
                          range(depth)]
            self.head = KL.SeparableConv2D(filters=num_anchors * num_values,
                                               name=f'{self.name}/box-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [KL.Conv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in range(depth)]
            self.head = KL.Conv2D(filters=num_anchors * num_values, name=f'{self.name}/box-predict', **options)
        self.bns = [
            [KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/box-{i}-bn-{j}') for j in
             range(3, 8)]
            for i in range(depth)]
        # self.bns = [[BatchNormalization(freeze=freeze_bn, name=f'{self.name}/box-{i}-bn-{j}') for j in range(3, 8)]
        #             for i in range(depth)]
        self.relu = KL.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = KL.Reshape((-1, num_values))
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        self.level += 1
        return outputs


class ClassNet(KL.Layer):
    def __init__(self, width, depth, num_classes=20, num_anchors=9, separable_conv=True, freeze_bn=False, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }
        if self.separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [KL.SeparableConv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                                 **options)
                          for i in range(depth)]
            self.head = KL.SeparableConv2D(filters=num_classes * num_anchors,
                                               bias_initializer=PriorProbability(probability=0.01),
                                               name=f'{self.name}/class-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [KL.Conv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                        **options)
                          for i in range(depth)]
            self.head = KL.Conv2D(filters=num_classes * num_anchors,
                                      bias_initializer=PriorProbability(probability=0.01),
                                      name='class-predict', **options)
        self.bns = [
            [KL.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/class-{i}-bn-{j}') for j
             in range(3, 8)]
            for i in range(depth)]
        # self.bns = [[BatchNormalization(freeze=freeze_bn, name=f'{self.name}/class-{i}-bn-{j}') for j in range(3, 8)]
        #             for i in range(depth)]
        self.relu = KL.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = KL.Reshape((-1, num_classes))
        self.activation = KL.Activation('sigmoid')
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation(outputs)
        self.level += 1
        return outputs

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)

def clip_boxes_graph(boxes, window):
    """
    Clip the normalized boxes coordinate to the range of [0,1]
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


class ROIAlignLayer(KL.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]
    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(ROIAlignLayer, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

  

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        #image_shape = inputs[1]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        #image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        image_area = tf.cast(inputs[1] * inputs[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)
            print("box at level", level_boxes.shape)
            print(feature_maps)
            print("feature_map size", feature_maps[i])
            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


def build_bbox_target(annotations_group, num_classes, image_size,
                      detect_quadrangle=False, anchor_parameters=ANCHOR_PARAMETERS):
    '''
    Build classification and regression targets for bboxes.
    Args:
        annotations_group: List(dict), a list of dictionaries contain the ground truth labels 
    Returns:

    '''
    anchors = anchors_for_shape((image_size, image_size), anchor_params=anchor_parameters)
    num_anchors = anchor_parameters.num_anchors()

    batches_targets = anchor_targets_bbox(
            anchors,
            image_group,
            annotations_group,
            num_classes=13,
            detect_quadrangle=detect_quadrangle
        )
    return list(batches_targets)


def build_mask_target_graph(rois, 
                            b_gt_boxes, 
                            b_gt_masks,
                            gt_class_ids,
                            negative_overlap=0.4, 
                            positive_overlap=0.5,
                            num_classes=13,
                            mask_shape=[28,28]):
    '''
    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    masks: [batch, TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.
    Note: Returned arrays might be zero padded if not enough target ROIs.
    '''
    print("b_gt_boxes", b_gt_boxes.shape)

    b_gt_boxes, non_zeros = trim_zeros_graph(b_gt_boxes, name='trim_gt_boxes')
    b_gt_masks = tf.gather(b_gt_masks, tf.where(non_zeros)[:, 0], axis=0,
                        name="trim_gt_masks")   
    # skip handle coco crowd for now
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    overlaps = overlaps_graph(rois, b_gt_boxes)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= positive_overlap)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where((roi_iou_max < negative_overlap))[:, 0]

    positive_count = int(TRAIN_ROIS_PER_IMAGE * 0.33)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / 0.33
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )

    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(b_gt_masks, -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)
    boxes = tf.gather(rois, positive_indices)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    b_masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                    box_ids,
                                    mask_shape)
    b_masks = tf.round(tf.squeeze(b_masks, axis=-1))
    positive_rois = tf.gather(rois, positive_indices)
    negative_rois = tf.gather(rois, negative_indices)
    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])   
    b_masks = tf.pad(b_masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, b_masks

class DetectionTargetLayer(KL.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.
    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type
    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        #self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        print(proposals.shape, gt_class_ids.shape, gt_boxes.shape, gt_masks.shape)

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_mask"]
        outputs = batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: build_mask_target_graph(
                w, y, z, x),
            1, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, TRAIN_ROIS_PER_IMAGE, 28, 28) # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]

def data_generator(dataset, config, shuffle=True, phi=0,
                   batch_size=1, mask_shape=[28, 28]):
    '''
    '''
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    num_classes = dataset.num_classes()

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            group=[]            
            
            while b < batch_size:
                # Increment index to pick next image. Shuffle if at the start of an epoch.
                image_index = (image_index + 1) % len(image_ids)
                group.append(image_index)
                b+=1

            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            images, annotations_group = dataset.compute_inputs_targets(group)
            
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(annotations_group['labels'] > 0):
                continue

            # first stage bbox Targets
            image_size = image_sizes[phi]
            outs = build_bbox_target(annotations_group, num_classes, image_size, detect_quadrangle=False)
            targets = []

            # mask targets [batch, max_instances, 28, 28] and second state bbox targets [batch, max_instaces, 4]
            masks_batch = np.zeros((batch_size, MAX_INSTANCES, mask_shape[0], mask_shape[1]), dtype=np.float32)
            boxes_batch = np.zeros((batch_size, MAX_INSTANCES, 4), dtype=np.float32)
            labels_batch = np.zeros((batch_size, MAX_INSTANCES), dtype=np.float32)
            for index, annotations in enumerate(annotations_group):
                gt_masks = annotations['masks']
                gt_boxes = annotations['bboxes']
                gt_labels = annotations['labels']
                # TODO: normalize gt_boxes np
                masks_batch[index, :gt_masks.shape[0], :, :] = gt_masks
                boxes_batch[index, :gt_boxes.shape[0], :] = gt_boxes
                labels_batch[index, :gt_labels.shape[0], :] = gt_labels
            targets.extend([masks_batch, boxes_batch, labels_batch])


            inputs = images.extend(outs)
            outputs = targets
             
            yield inputs, outputs

            # start a new batch
            b = 0
        
        except (GeneratorExit, KeyboardInterrupt):
            raise


def build_fpn_mask_graph(rois, feature_maps, image_size, num_classes,
                         pool_size, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P1, P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = ROIAlignLayer([pool_size, pool_size],
                        name="roi_align_mask")((rois, image_size, feature_maps))

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), 
                           name='mask')(x)
    #print("pool mask feature map size", x.shape)
    return x


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


class EfficientDetModel():
    def __init__(self, phi):
        assert phi in range(7)
        self.phi = phi
        self.input_size = image_sizes[phi]
        self.input_shape = (self.input_size, self.input_size, 3)
        self.w_bifpn = w_bifpns[phi]
        self.d_bifpn = d_bifpns[phi]
        self.w_head = self.w_bifpn
        self.d_head = d_heads[phi]
        self.backbone_cls = backbones[phi]
        self.model = self.efficientdet()

    def efficientdet(self, num_classes=13, num_anchors=9, weighted_bifpn=False, freeze_bn=False,
                    score_threshold=0.01, detect_quadrangle=False, anchor_parameters=None, separable_conv=True,
                    mask_shape=[28,28]):
        image_input = KL.Input(self.input_shape)
        '''
        w_bifpn = w_bifpns[phi]
        d_bifpn = d_bifpns[phi]
        w_head = w_bifpn
        d_head = d_heads[phi]
        '''
        features = self.backbone_cls(input_tensor=image_input, freeze_bn=freeze_bn)
        if weighted_bifpn:
            fpn_features = features
            for i in range(self.d_bifpn):
                fpn_features = build_wBiFPN(fpn_features, self.w_bifpn, i, freeze_bn=freeze_bn)
        else:
            fpn_features = features
            for i in range(self.d_bifpn):
                fpn_features = build_BiFPN(fpn_features, self.w_bifpn, i, freeze_bn=freeze_bn)
        box_net = BoxNet(self.w_head, self.d_head, num_anchors=num_anchors, separable_conv=separable_conv, freeze_bn=freeze_bn,
                        detect_quadrangle=detect_quadrangle, name='box_net')
        class_net = ClassNet(self.w_head, self.d_head, num_classes=num_classes, num_anchors=num_anchors,
                            separable_conv=separable_conv, freeze_bn=freeze_bn, name='class_net')
        classification = [class_net([feature, i]) for i, feature in enumerate(fpn_features)]
        classification = KL.Concatenate(axis=1, name='classification')(classification)
        regression = [box_net([feature, i]) for i, feature in enumerate(fpn_features)]
        regression = KL.Concatenate(axis=1, name='regression')(regression)
        # regression has shpae [batch, num_detections, 4]

        # apply predicted regression to anchors
        anchors = anchors_for_shape((self.input_size, self.input_size), anchor_params=anchor_parameters)
        anchors_input = np.expand_dims(anchors, axis=0)

        # build input layer for box detection targets
        box_target = KL.Input((anchors.shape[0], 5), name="box_target")
        class_target = KL.Input((anchors.shape[0], num_classes+1), name="class_target")

        boxes = RegressBoxes(name='boxes')([anchors_input, regression[..., :4]])
        boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])
        
        # apply nms on boxes???
        # predict masks base on boxes
        # TODO: normalize regression, nms on regression
        roi_masks = build_fpn_mask_graph(regression, fpn_features, self.input_size, 13, 7) #[batch, num_proposals,h,w,num_classes]
        # TODO: generate target TRAIN_ROIS_PER_IMAGE rois 
        #target_rois = .. shape[batch, num_rois, 4]
        target_rois = KL.Input((100, 4))

        gt_masks = KL.Input((MAX_INSTANCES, mask_shape[0], mask_shape[1]))
        gt_boxes = KL.Input((MAX_INSTANCES, 4))
        gt_labels = KL.Input((MAX_INSTANCES))

        rois, target_class_ids, target_mask = DetectionTargetLayer(name="proposal_targets")([
                    target_rois, gt_labels, gt_boxes, gt_masks])

        # build losses
        output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

        # Losses
        regression_loss = KL.Lambda(lambda x: focal(*x), name="regression_loss")([box_target, regression])     
        classification_loss = KL.Lambda(lambda x: smooth_l1(*x), name="classification_loss")([class_target, classification]) 
        mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                                        [target_mask, target_class_ids, roi_masks])  


        # filter detections (apply NMS / score threshold / select top-k)
        if detect_quadrangle:
            detections = FilterDetections(
                name='filtered_detections',
                score_threshold=score_threshold,
                detect_quadrangle=True
            )([boxes, classification, regression[..., 4:8], regression[..., 8]])
        else:
            detections = FilterDetections(
                name='filtered_detections',
                score_threshold=score_threshold
            )([boxes, classification])

        model = models.Model(inputs=[image_input, box_target, class_target], 
                             outputs=[classification, regression,roi_masks], name='efficientdet')
        prediction_model = models.Model(inputs=[image_input, box_target, class_target], 
                                        outputs=detections, name='efficientdet_p')
        return model, prediction_model

    def compile(self,learning_rate=1e-3):
        self.model.compile(optimizer=Adam(lr=1e-3))
        # compile losses
        self.model._losses = []
        self.model._per_input_losses = {}
        loss_names = [
            "regression_loss", "classification_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.model.get_layer(name)
            if layer.output in self.model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True))
            self.model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        '''
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))'''

        # Compile
        self.model.compile(
            optimizer=Adam(lr=1e-3),
            loss=[None] * len(self.model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.model.metrics_names:
                continue
            layer = self.model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True))
            self.model.metrics_tensors.append(loss)


    def create_callbacks(self, training_model, prediction_model, validation_generator, args):
        """
        Creates the callbacks to use during training.

        Args
            training_model: The model that is used for training.
            prediction_model: The model that should be used for validation.
            validation_generator: The generator for creating validation data.
            args: parseargs args object.

        Returns:
            A list of callbacks used for training.
        """
        callbacks = []

        tensorboard_callback = None

        if args.tensorboard_dir:
            if tf.version.VERSION > '2.0.0':
                file_writer = tf.summary.create_file_writer(args.tensorboard_dir)
                file_writer.set_as_default()
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=args.tensorboard_dir,
                histogram_freq=0,
                batch_size=args.batch_size,
                write_graph=True,
                write_grads=False,
                write_images=False,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None
            )
            callbacks.append(tensorboard_callback)

        if args.evaluation and validation_generator:
            if args.dataset_type == 'coco':
                from eval.coco import Evaluate
                # use prediction model for evaluation
                evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
            else:
                from eval.pascal import Evaluate
                evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
            callbacks.append(evaluation)

        # save the model
        if args.snapshots:
            # ensure directory created first; otherwise h5py will error after epoch.
            makedirs(args.snapshot_path)
            checkpoint = keras.callbacks.ModelCheckpoint(
                os.path.join(
                    args.snapshot_path,
                    f'{args.dataset_type}_{{epoch:02d}}_{{loss:.4f}}_{{val_loss:.4f}}.h5' if args.compute_val_loss
                    else f'{args.dataset_type}_{{epoch:02d}}_{{loss:.4f}}.h5'
                ),
                verbose=1,
                save_weights_only=True,
                # save_best_only=True,
                # monitor="mAP",
                # mode='max'
            )
            callbacks.append(checkpoint)

        # callbacks.append(keras.callbacks.ReduceLROnPlateau(
        #     monitor='loss',
        #     factor=0.1,
        #     patience=2,
        #     verbose=1,
        #     mode='auto',
        #     min_delta=0.0001,
        #     cooldown=0,
        #     min_lr=0
        # ))

        return callbacks

    def train(self, train_dataset, val_dataset, learning_rate, epochs,
              config, custom_callbacks=None):
        # Data generators
        train_generator = data_generator(train_dataset, config, shuffle=True,
                                         phi=self.phi, batch_size=1)
        val_generator = data_generator(val_dataset, config, shuffle=True,
                                       batch_size=1, phi=self.phi)

        # Create log_dir if it does not exist
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=config.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(os.path.join(
                                            config.checkpoint_path,
                                            f'{args.dataset_type}_{{epoch:02d}}_{{loss:.4f}}_{{val_loss:.4f}}.h5' if args.compute_val_loss
                                            else f'{args.dataset_type}_{{epoch:02d}}_{{loss:.4f}}.h5'),
                                            verbose=0, save_weights_only=True)
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.compile()
        #self.set_trainable(layers)
        #self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

if __name__ == '__main__':
    model = EfficientDetModel(0)
    x,y = model.efficientdet()
    #x.train()
    
