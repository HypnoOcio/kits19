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
# using Restnet 50/101 as encoder
# adjusted sizes of CNN filters 
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
        
        new_height, new_width = self.new_size
        #resized = tf.image.resize_nearest_neighbor(inputs, self.new_size)
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
#(473,473,3)
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
    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0], use_bias=False)(inp)   
    bn1 = BN(name=names[1])(cnv1)         
    relu1 = Activation('relu')(bn1)       
    # "conv1_2_3x3"
    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2], use_bias=False)(relu1) 
    bn1 = BN(name=names[3])(cnv1)        
    relu1 = Activation('relu')(bn1)       
    # "conv1_3_3x3"
    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],use_bias=False)(relu1) 
    bn1 = BN(name=names[5])(cnv1)         
    relu1 = Activation('relu')(bn1)       
    # "pool1_3x3_s2"
    res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(relu1)                     

    # Residual layers(body of network)
    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """
    
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)                                           
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)                                   
    
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)                      
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)                                   
    
    if layers is 50:
                                                                    
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)                                                 
        for i in range(5):
            
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)                                    
    
    elif layers is 101:     
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)                                  
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")
    
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)                                           
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)                                   

    res = Activation('relu')(res)
    return res


def interp_block(prev_layer, level, feature_map_shape, input_shape, regularizer):
    input_shape = (input_shape[0],input_shape[1])
    kernel_strides_map= {}
    if input_shape == (128, 128):
        kernel_strides_map = {1: 16, 2: 8, 4: 4, 8: 2}
        #kernel_strides_map = {1: 32, 2: 16, 4: 8, 8: 4}
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
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0], kernel_regularizer = regularizer, use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    prev_layer = Interp(feature_map_shape)(prev_layer)

    return prev_layer


#input_shape=(473,473)，res(60,60,2048)
def build_pyramid_pooling_module(res, input_shape, regularizer):
    """Build the Pyramid Pooling Module."""
    feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape)
    
    #1: res1x1x2048, 512, 60x60x512
    interp_block1 = interp_block(res, 1, feature_map_size, input_shape, regularizer)                  # (60,60，512)
    #2：res2x2x2048，512，60x60x512
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape, regularizer)                  # (60,60，512)
    #3：res3x3x2048，512，60x60x512
    interp_block3 = interp_block(res, 4, feature_map_size, input_shape, regularizer)                  # (60,60，512)
    #4：res6x6x2048，512，60x60x512
    interp_block6 = interp_block(res, 8, feature_map_size, input_shape, regularizer)                  # (60,60，512)

    # (60,60,(512x4+2048))=(60,60,4096)
    res = Concatenate()([res,interp_block6,interp_block3,interp_block2,interp_block1])                  #(60,60,4096)

    return res


def pspnet_big(num_classes, input_shape , resnet_layers=50, regularizer = None):
    """Build PSPNet."""
    print("Building a PSPNet based on ResNet %i expecting inputs of shape %s and predicting %i classes" % (
        resnet_layers, input_shape, num_classes))

    inp = Input((input_shape[0], input_shape[1], input_shape[2]))                          # (128,128,3)
    input_shape = (input_shape[0],input_shape[1])

    #ResNet
    res = ResNet(inp, layers=resnet_layers)                                              # (60,60,2048)
    
    print(input_shape)
    psp = build_pyramid_pooling_module(res, input_shape, regularizer)                      # (60,60,4096)

    # 1x1，512
    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4", kernel_regularizer = regularizer,   # (60,60,512)
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # 1x1，num_class
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), name="conv6")(x)           # (60,60,num_class)

    
    x = Interp([input_shape[0], input_shape[1]])(x)                           # (128,128,num_class)
    x = Activation('softmax')(x)

    model = Model(inputs=inp, outputs=x)

    return model