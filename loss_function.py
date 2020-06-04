import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K
from constants import *
from keras.losses import categorical_crossentropy

# source - #https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return K.mean( (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) )


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice


def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))

# y_true[..., 0] is RED - kidney
# y_true[..., 2] is BLUE - tumor
def double_head_loss(y_true, y_pred):
    kidney_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])
    tumor_loss = dice_coef_loss_bce(y_true[..., 2], y_pred[..., 2])
    return kidney_loss + tumor_loss

def kidney_tumor_loss(y_true, y_pred):
    kidney_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])

    tumor_loss = dice_coef_loss_bce(y_true[..., 2], y_pred[..., 2])
    full_mask = dice_coef_loss_bce(y_true, y_pred)
    kidney_loss = K.sum(kidney_loss)
    tumor_loss = K.sum(tumor_loss)
    full_mask = K.sum(full_mask)
    return (3/8) * kidney_loss + (4/8) * tumor_loss + (1/8) * full_mask 


def softmax_dice_loss(y_true, y_pred, alpha = 0.5, beta = 0.5):
    return alpha * categorical_crossentropy(y_true, y_pred) + beta * dice_coef_loss(y_true, y_pred)


def softmax_dice_loss_tune_params(alpha, beta):
    def sdl(y_true, y_pred):
        return alpha * categorical_crossentropy(y_true, y_pred) + beta * dice_coef_loss(y_true, y_pred)
    return sdl

#-----------------------------------------------------#
#                    Tversky loss                     #
#-----------------------------------------------------#
#                     Reference:                      #
#                Sadegh et al. (2017)                 #
#     Tversky loss function for image segmentation    #
#      using 3D fully convolutional deep networks     #
#-----------------------------------------------------#
# alpha=beta=0.5 : dice coefficient                   #
# alpha=beta=1   : jaccard                            #
# alpha+beta=1   : produces set of F*-scores          #
#-----------------------------------------------------#
def tversky_loss(y_true, y_pred, smooth=0.000001):
    # Define alpha and beta
    alpha = 0.5
    beta  = 0.5
    # Calculate Tversky for each class
    tp = K.sum(y_true * y_pred, axis=[1,2,3])
    fn = K.sum(y_true * (1-y_pred), axis=[1,2,3])
    fp = K.sum((1-y_true) * y_pred, axis=[1,2,3])
    tversky_class = (tp + smooth)/(tp + alpha*fn + beta*fp + smooth)
    # Sum up classes to one score
    tversky = K.sum(tversky_class, axis=[-1])
    # Identify number of classes
    n = K.cast(K.shape(y_true)[-1], 'float32')
    # Return Tversky
    return n-tversky

def tversky_adjusted(y_true, y_pred, alpha, beta, class_weights = None, smooth=0.000001):
    if class_weights is not None:
      class_weights = K.constant(np.array(class_weights))

    # element - wise multiplication -> more precise metrics because background
    # has smaller weights then kidney and tumor
    if class_weights is not None:
      tp = K.sum(y_true * y_pred * class_weights )
      fn = K.sum(y_true * (1-y_pred) * class_weights )
      fp = K.sum((1-y_true) * y_pred * class_weights )
    else:
      y_true = K.flatten(y_true)
      y_pred = K.flatten(y_pred)
      tp = K.sum(y_true * y_pred )
      fn = K.sum(y_true * (1-y_pred) )
      fp = K.sum((1-y_true) * y_pred )

    tversky = (tp + smooth)/(tp + alpha*fn + beta*fp + smooth)

    return tversky

# bigger beta[0,inf]; beta > 1 returns more false positives
def weighted_cross_entropy(beta):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)
    loss.__name__= 'weighted_CE'
    return loss

#beta[0,1] beta < 1; beta closer to 0 penalizes more negative examples
def balanced_cross_entropy(beta):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))
    loss.__name__='balanced_CE'

    return loss

def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.log(y_pred / (1 - y_pred))
        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    loss.__name__ = 'focal_loss'

    return loss


def make_loss(loss_name,alpha,beta,class_weights = None):
    if loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.5, bce=0.5)
        return loss

    elif loss_name == 'softmax_dice_loss':
        def loss(y, p):
            return softmax_dice_loss(y, p, alpha=alpha, beta=beta)
        loss.__name__ = 'softmax_dice_loss'
        return loss

    elif loss_name == 'tversky' or loss_name == 'tversky_loss':
        def loss(y, p):
            if loss_name == 'tversky':   
                # return tversky index
                return tversky_adjusted(y, p, alpha = alpha, beta = beta, class_weights = class_weights)
                # return 1 - (tversky index) = loss
            return 1 - tversky_adjusted(y, p, alpha = alpha, beta = beta, class_weights = class_weights)

        if alpha == 0.5 and beta == 0.5:
            loss.__name__ = 'dice_coef'
        elif alpha == 1.0 and beta == 1.0:
            loss.__name__ = 'jacc'

        if class_weights is not None:
            loss.__name__ += "_w"

        if loss_name == 'tversky_loss':
            loss.__name__ += '_loss'

        return loss

    elif loss_name == 'bce':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0, bce=1)
        return loss

    elif loss_name == 'weighted_cross_entropy':
        return weighted_cross_entropy(beta = beta)
    elif loss_name == 'balanced_cross_entropy':
        return balanced_cross_entropy(beta = beta)
    elif loss_name == 'focal_loss':
        return focal_loss(alpha=alpha, gamma=beta)

    elif loss_name == 'categorical_dice':
        return softmax_dice_loss
    elif loss_name == 'double_head_loss':
        return double_head_loss
    elif loss_name == 'kidney_tumor_loss':
        return kidney_tumor_loss
    else:
        ValueError("Unknown loss.")
