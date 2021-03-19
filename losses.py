"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import keras
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K


def focal(y_true, y_pred, alpha=0.25, gamma=1.5):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

   
    """
    Compute the focal loss given the target tensor and the predicted tensor.

    As defined in https://arxiv.org/abs/1708.02002

    Args
        y_true: Tensor of target data from the generator with shape (B, N, num_classes).
        y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

    Returns
        The focal loss of y_pred w.r.t. y_true.
    """
    labels = y_true[:, :, :-1]
    # -1 for ignore, 0 for background, 1 for object
    anchor_state = y_true[:, :, -1]
    classification = y_pred

    # filter out "ignore" anchors
    indices = tf.where(keras.backend.not_equal(anchor_state, -1))
    labels = tf.gather_nd(labels, indices)
    classification = tf.gather_nd(classification, indices)

    # compute the focal loss
    alpha_factor = keras.backend.ones_like(labels) * alpha
    alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
    focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
    focal_weight = alpha_factor * focal_weight ** gamma
    cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.where(keras.backend.equal(anchor_state, 1))
    normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
    normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

    return keras.backend.sum(cls_loss) / normalizer



def smooth_l1(y_true, y_pred, sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    
    """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
    Args
        y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
        y_pred: Tensor from the network of shape (B, N, 4).
    Returns
        The smooth L1 loss of y_pred w.r.t. y_true.
    """
    # separate target and state
    regression = y_pred
    regression_target = y_true[:, :, :-1]
    anchor_state = y_true[:, :, -1]

    # filter out "ignore" anchors
    indices = tf.where(keras.backend.equal(anchor_state, 1))
    regression = tf.gather_nd(regression, indices)
    regression_target = tf.gather_nd(regression_target, indices)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = regression - regression_target
    regression_diff = keras.backend.abs(regression_diff)
    regression_loss = tf.where(
        keras.backend.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    # compute the normalizer: the number of positive anchors
    normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
    normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
    return keras.backend.sum(regression_loss) / normalizer



def smooth_l1_quad(sigma=3.0):
    """
    Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        print(y_pred.shape, y_true.shape)
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression = tf.concat([regression[..., :4], tf.sigmoid(regression[..., 4:9])], axis=-1)
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        box_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., :4], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., :4], 2),
            regression_diff[..., :4] - 0.5 / sigma_squared
        )

        alpha_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., 4:8], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., 4:8], 2),
            regression_diff[..., 4:8] - 0.5 / sigma_squared
        )

        ratio_regression_loss = tf.where(
            keras.backend.less(regression_diff[..., 8], 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff[..., 8], 2),
            regression_diff[..., 8] - 0.5 / sigma_squared
        )
        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())

        box_regression_loss = tf.reduce_sum(box_regression_loss) / normalizer
        alpha_regression_loss = tf.reduce_sum(alpha_regression_loss) / normalizer
        ratio_regression_loss = tf.reduce_sum(ratio_regression_loss) / normalizer

        return box_regression_loss + alpha_regression_loss + 16 * ratio_regression_loss

    return _smooth_l1

def mask_loss():

    def _mask_loss(y_true, y_pred):
        print("in here")
        """Mask binary cross-entropy loss for the masks head.
        target_masks: [batch, num_rois, height, width].
            A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                    with values from 0 to 1.
        """
        y_pred = tf.squeeze(y_pred, axis=-1)
        #print(y_pred.shape, y_true.shape)
        #assert y_pred.shape == y_true.shape, "Shape mismatch"
        # Compute binary cross entropy. If no positive ROIs, then return 0.
        # shape: [batch, roi, num_classes]
        loss = K.switch(tf.size(y_true) > 0,
                        K.binary_crossentropy(target=y_true, output=y_pred),
                        tf.constant(0.0))
        loss = K.mean(loss)
        return loss
    
    return _mask_loss
