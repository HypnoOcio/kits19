# source for model https://gist.github.com/EternalSorrrow/f8af26a007b23ea32a50f250813a82e7

import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, ZeroPadding2D, concatenate, add, SpatialDropout2D
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

#Better model, contatining less parameters and converging faster
def dense_conv_block(x, growth_rate, name, use_bias, regularizer):
  x1 = BatchNormalization(name=name+'_bn')(x)
  x1 = Activation('relu', name=name+'_relu')(x1)
  x1 = Conv2D(growth_rate, (3,3), name=name+'_conv', padding='same',
              use_bias=use_bias, kernel_regularizer=regularizer)(x1)
  return concatenate([x, x1])

def dense_block(x, conv_blocks, growth_rate, name, use_bias, regularizer):
  for i in range(conv_blocks):
    block_name = name + '_' + str(i)
    x = dense_conv_block(x, growth_rate, name=block_name, use_bias=use_bias, regularizer=regularizer)
  return x

def transition_down(x):
  x = MaxPooling2D((2,2))(x)
  return x

def transition_up(x, name, reduction, use_bias, regularizer):
  x = BatchNormalization(name=name+'_bn')(x)
  x = Activation('relu')(x)
  x = Conv2D(int(K.int_shape(x)[3] * reduction), 1, name = name + '_1x1conv',
              use_bias=use_bias, kernel_regularizer=regularizer)(x)
  x = UpSampling2D((2, 2))(x)

  return x

def dense_stem(x, filters, name, use_bias, regularizer):
  x = Conv2D(filters, (3,3), padding='same', name=name+'_conv',
              use_bias = use_bias, kernel_regularizer = regularizer)(x)
  x = BatchNormalization(name = name + '_bn')(x)
  x = Activation('relu')(x)
  return x

def reduction(x, reduction, name, use_bias, regularizer):
  x = BatchNormalization(name = name + '_bn')(x)
  x = Activation('relu')(x)
  x = Conv2D(int(K.int_shape(x)[3] * reduction), (1,1), name=name+'_1x1conv',
              use_bias = use_bias, kernel_regularizer = regularizer)(x)
  return x

def Dense_UNet_small(model_input,num_classes, dropout=0.05, use_bias=True, regularizer=None):
  blocks = [4, 4, 4, 4, 4] #[12, 12, 12, 12, 12]
  growth_rate = [4, 8, 16, 32, 64]

  #Contracting path
  x = dense_stem(model_input, 16, 'stem', use_bias, regularizer)

  d1 = dense_block(x, blocks[0], growth_rate[0], 'dense1', use_bias, regularizer)
  p1 = transition_down(d1)
  p1 = SpatialDropout2D(dropout)(p1)

  d2 = dense_block(p1, blocks[1], growth_rate[1], 'dense2', use_bias, regularizer)
  p2 = transition_down(d2)
  p2 = SpatialDropout2D(dropout)(p2)

  d3 = dense_block(p2, blocks[2], growth_rate[2], 'dense3', use_bias, regularizer)
  p3 = transition_down(d3)
  p3 = SpatialDropout2D(dropout)(p3)

  d4 = dense_block(p3, blocks[3], growth_rate[3], 'dense4', use_bias, regularizer)
  p4 = transition_down(d4)
  p4 = SpatialDropout2D(dropout)(p4)

  d5 = dense_block(p4, blocks[4], growth_rate[4], 'dense5', use_bias, regularizer)
  d5 = SpatialDropout2D(dropout)(d5)

  #Expanding path\n",
  u1 = transition_up(d5, 'up1', 0.5, use_bias, regularizer)
  c1 = concatenate([u1, d4])
  c1 = SpatialDropout2D(dropout)(c1)
  r1 = reduction(c1, 0.25, 'reduction1', use_bias, regularizer)
  d6 = dense_block(r1, blocks[3], growth_rate[3], 'dense6', use_bias, regularizer)

  u2 = transition_up(d6, 'up2', 0.5, use_bias, regularizer)
  c2 = concatenate([u2, d3])
  c2= SpatialDropout2D(dropout)(c2)
  r2 = reduction(c2, 0.25, 'reduction2', use_bias, regularizer)
  d7 = dense_block(r2, blocks[2], growth_rate[2], 'dense7', use_bias, regularizer)


  u3 = transition_up(d7, 'up3', 0.5, use_bias, regularizer)
  c3 = concatenate([u3, d2])
  c3 = SpatialDropout2D(dropout)(c3)
  r3 = reduction(c3, 0.25, 'reduction3', use_bias, regularizer)
  d8 = dense_block(r3, blocks[1], growth_rate[1], 'dense8', use_bias, regularizer)


  u4 = transition_up(d8, 'up4', 0.5, use_bias, regularizer)
  c4 = concatenate([u4, d1])
  c4 = SpatialDropout2D(dropout)(c4)
  r4 = reduction(c4, 0.25, 'reduction4', use_bias, regularizer)
  d9 = dense_block(r4, blocks[0], growth_rate[0], 'dense9', use_bias, regularizer)

  r5 = reduction(d9, 0.25, 'reduction5', use_bias, regularizer)
  c5 = concatenate([r5, x])
  d10 = dense_block(r4, blocks[0], growth_rate[0], 'dense10', use_bias, regularizer)

  outputs = Conv2D(num_classes, 1, activation='softmax', name='output')(d10)
  model = Model(model_input, outputs)

  return model