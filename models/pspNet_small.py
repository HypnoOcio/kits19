#%tensorflow_version 1.x
from __future__ import print_function
from math import ceil
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K
import tensorflow as tf

# adjusted number of psp regions and size of filters
# net is proportionally adjusted to other compared nets ( 1-2 million parameters )
# added sizes of filters in build_pyramid_pooling_module and for ResNet so net is easily to reuse 
# using customized smaller encoder instead of Resnet50/101
# adjusted sizes of CNN filters so images 
# added regularization


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


class Interp(layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # resized = tf.image.resize_nearest_neighbor(inputs, self.new_size)
        new_height, new_width = self.new_size
        resized = tf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
         return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config

def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0], use_bias=False)(prev)
    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad, pad))(prev)
    prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad, name=names[2], use_bias=False)(prev)
    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)

    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4], use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj", "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0], use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev

def empty_branch(prev):
    return prev

 
def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added

def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)

    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added


# ResNet
# input(473,473,3) - size of dimensions for this input
def ResNet(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)
    # "conv1_1_3x3_s2"
    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0], use_bias=False)(inp)   # 237x237x64
    bn1 = BN(name=names[1])(cnv1)         
    relu1 = Activation('relu')(bn1)       
    # "conv1_2_3x3"
    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2], use_bias=False)(relu1) # 237x237x64
    bn1 = BN(name=names[3])(cnv1)        
    relu1 = Activation('relu')(bn1)       
    # "conv1_3_3x3"
    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],use_bias=False)(relu1) # 237x237x128
    bn1 = BN(name=names[5])(cnv1)         
    relu1 = Activation('relu')(bn1)       
    # "pool1_3x3_s2"
    res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(relu1)                     # 119x119x128

    # Residual layers(body of network)
    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """
    # conv2_x,ResNet50/101,conv2_1-conv2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)                                           # 119x119x256
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)                                   # 119x119x256
    
    # conv3_x,ResNet50/101，conv3_1-conv3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)                       # 60x60x512
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)                                   # 60x60x512
    
    # conv4_x,ResNet50/101
    # ResNet50，conv4_1 - conv4_6
    # conv4_1x1 ,conv4_3x3
    if layers is 50:
        # conv4_1                                                              
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)                                       # 60x60x1024            
        for i in range(5):
            # conv4_2-conv4_6_3x3
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)                               # 60x60x1024       
    # ResNet101，conv4_1 - conv4_23
    # cconv4_3x3
    elif layers is 101:
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)    
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")
    
    # conv5_x,ResNet50/101，conv5_1 - conv5_3
    
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)                                           # 60x60x2048
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)                                   # 60x60x2048

    res = Activation('relu')(res)
    return res


# input_shape(128,128)
def interp_block(prev_layer, level, feature_map_shape, input_shape, regularizer):
    kernel_strides_map= {}
    if input_shape == (128, 128):
        #we use regions of such size
        kernel_strides_map = {1: 16, 2: 8, 4: 4, 8: 2}
        #kernel_strides_map = {1: 32, 2: 16, 4: 8, 8: 4}
    elif input_shape == (473, 473):
        kernel_strides_map = {1: 60, 2: 30, 3: 20, 6: 10}
    elif input_shape == (713, 713):
        kernel_strides_map = {1: 90, 2: 45, 3: 30, 6: 15}
    elif input_shape == (320,320):                                  
        kernel_strides_map = {1: 40, 2: 20, 4: 10, 8: 5}        
    elif input_shape == (512, 512):
        kernel_strides_map = {1: 64, 2: 32, 4: 16, 8: 8}
    else:
        print("Pooling parameters for input shape ", input_shape, " are not defined.")
        exit(1)

    names = ["conv5_3_pool" + str(level) + "_conv", 
             "conv5_3_pool" + str(level) + "_conv_bn"]

    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    
    prev_layer = Conv2D(256, (1, 1), strides=(1, 1), kernel_regularizer = regularizer, name=names[0], use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    
    #(60,60,512)
    prev_layer = Interp(feature_map_shape)(prev_layer)

    return prev_layer


# input_shape=(128,128,3)
def build_pyramid_pooling_module(cnn_out, inp, input_shape, regularizer):
    """Build the Pyramid Pooling Module."""

    # feature_map_size = (32,32)
    feature_map_size = tuple(int(ceil(input_dim / 4.0)) for input_dim in input_shape)
    
    res = inp
    
    interp_block1 = interp_block(res, 1, feature_map_size, input_shape, regularizer)                  # out shape(32,32,256)
    
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape, regularizer)                  # out shape(32,32,256)
    
    interp_block3 = interp_block(res, 4, feature_map_size, input_shape, regularizer)                  # out shape(32,32,256)
    
    interp_block6 = interp_block(res, 8, feature_map_size, input_shape, regularizer)                  # out shape(32,32,256)

    #(32,32,(256*4+256))=(60,60,1280)
    res = Concatenate()([cnn_out,interp_block6,interp_block3,interp_block2,interp_block1])   

    return res


# shape input (128,128,3)
def my_conv_preprocess(inp,regularizer):
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(inp)
    conv1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #(64,64,64)
    #conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(conv2)
    conv3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    #(32,32,256)
    return pool3


#input shape (128,128,3)
def pspnet_small(num_classes, input_shape , resnet_layers=50, regularizer = None):
    """Build PSPNet."""
    print("Building a PSPNet based on custom encoder expecting inputs of shape %s and predicting %i classes" % (
         input_shape, num_classes))

    
    inp = Input((input_shape[0], input_shape[1], input_shape[2]))                      
    input_shape = (input_shape[0],input_shape[1])

    #replacement instead of Resnet - too many parameters 
    cnn_out = my_conv_preprocess(inp, regularizer)                                                       #out shape (32,32,256)
    psp = build_pyramid_pooling_module(cnn_out,inp,input_shape,regularizer)                 #out shape (32,32,1280)

    
    x = Conv2D(256, (2, 2), strides=(1, 1), padding="same", kernel_regularizer = regularizer, name="conv5_4",     #out shape (32,32,256)
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(num_classes, (1, 1), strides=(1, 1), name="conv6")(x)            # out shape(32,32,num_class)

    x = Interp([input_shape[0], input_shape[1]])(x)                             # out shape(128,128,num_class)
    x = Activation('softmax')(x)

    model = Model(inputs=inp, outputs=x)

    return model
