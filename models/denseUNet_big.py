#%tensorflow_version 1.x

import numpy as np
from keras.optimizers import SGD
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, ZeroPadding2D, concatenate, add
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

# added filters for DenseNet 121
# adjusted architecture of the model, number of filters and sizes so images of (128,128,3) would fit
# added regularization
# reduced number of learning parameters by adding 1x1 convolution, big enough not to be bottleneck

def Dense_UNet_big(input_shape, num_classes, nb_dense_block=4, growth_rate=48, nb_filter=96, regularizer = None, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, weights_path=None):
    '''Instantiate the DenseNet 161 architecture or DenseNet 121,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            num_classes: optional number of num_classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction
    global concat_axis
    concat_axis = 3
    img_input = Input(shape=(input_shape[0], input_shape[1], input_shape[2]), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    # nb_filter = 96
    # nb_layers = [6,12,36,24] # For DenseNet-161
    nb_filter = 64
    nb_layers = [6,12,24,16] # For DenseNet-121
    box = []
    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), kernel_initializer="he_normal", name='conv1', use_bias=False, kernel_regularizer=regularizer)(x)
    x = BatchNormalization(epsilon=eps, scale=True, axis=concat_axis, name='conv1_bn')(x)
    #x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    box.append(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, regularizer, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        box.append(x)
        # Add transition_block
        x = transition_block(x, stage, nb_filter, regularizer, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, regularizer, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps,scale=True, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    box.append(x)

    up0 = UpSampling2D(size=(2,2))(x)
    filter_num0 = 1024
    filter_num1 = 768
    filter_num2 = 384
    filter_num3 = 192

    #------
    up0 = Conv2D( filter_num0, (1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizer, name = "conv_up0_conv1x1")(up0)
    up0 = BatchNormalization(name = "bn_up0_conv1x1")(up0)
    up0 = Activation('relu', name='ac_up0_conv1x1')(up0)
    #------
    line0 = Conv2D(filter_num0, (1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizer, name="line0")(box[3])
    up0_sum = add([line0, up0])
    conv_up0 = Conv2D(filter_num1, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizer, name = "conv_up0")(up0_sum)
    bn_up0 = BatchNormalization(name = "bn_up0")(conv_up0)
    ac_up0 = Activation('relu', name='ac_up0')(bn_up0)
    up1 = UpSampling2D(size=(2,2))(ac_up0)
    #----
    box2 = Conv2D(filter_num1, (1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizer, name = "conv_up_box2_conv1x1")(box[2])
    bn_box2 = BatchNormalization(name = "bn_box2_conv1x1")(box2)
    ac_box2 = Activation('relu', name='ac_box2_conv1x1')(bn_box2)
    box2 = ac_box2
    #----
    up1_sum = add([box2, up1])
    conv_up1 = Conv2D(filter_num2, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizer, name = "conv_up1")(up1_sum)
    bn_up1 = BatchNormalization(name = "bn_up1")(conv_up1)
    ac_up1 = Activation('relu', name='ac_up1')(bn_up1)
    up2 = UpSampling2D(size=(2,2))(ac_up1)
    #----
    box1 = Conv2D(filter_num2, (1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizer, name = "conv_up_box1_conv1x1")(box[1])
    bn_box1 = BatchNormalization(name = "bn_box1_conv1x1")(box1)
    ac_box1 = Activation('relu', name='ac_box1_conv1x1')(bn_box1)
    box1 = ac_box1
    #----
    up2_sum = add([box1, up2])
    conv_up2 = Conv2D(filter_num3, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizer, name = "conv_up2")(up2_sum)
    bn_up2 = BatchNormalization(name = "bn_up2")(conv_up2)
    ac_up2 = Activation('relu', name='ac_up2')(bn_up2)
    up3 = UpSampling2D(size=(2,2))(ac_up2)
    #----
    box0 = Conv2D(filter_num3, (1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizer, name = "conv_up_box0_conv1x1")(box[0])
    bn_box0 = BatchNormalization(name = "bn_box0_conv1x1")(box0)
    ac_box0 = Activation('relu', name='ac_box0_conv1x1')(bn_box0)
    box0 = ac_box0
    #----
    up3_sum = add([box0, up3])
    conv_up3 = Conv2D(96, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizer, name = "conv_up3")(up3_sum)
    bn_up3 = BatchNormalization(name = "bn_up3")(conv_up3)
    ac_up3 = Activation('relu', name='ac_up3')(bn_up3)

    up4 = UpSampling2D(size=(2, 2))(ac_up3)
    conv_up4 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizer, name="conv_up4")(up4)
    conv_up4 = Dropout(rate=0.3)(conv_up4)
    bn_up4 = BatchNormalization(name="bn_up4")(conv_up4)
    ac_up4 = Activation('relu', name='ac_up4')(bn_up4)

    x = Conv2D(num_classes, (1,1),activation = "softmax", padding="same", kernel_initializer="he_normal", name="dense167classifer")(ac_up4)
    
    
    name_net = "denseu121" if nb_filter == 64 else "denseu161"
    model = Model(img_input, x, name=name_net)

    if weights_path is not None:
      model.load_weights(weights_path)

    return model


def conv_block(x, stage, branch, nb_filter, regularizer, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps,scale=True, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1),kernel_initializer="he_normal", name=conv_name_base+'_x1', use_bias=False, kernel_regularizer=regularizer )(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps,scale=True, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    #x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3),kernel_initializer="he_normal", name=conv_name_base+'_x2', use_bias=False, kernel_regularizer=regularizer)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter,regularizer, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps,scale=True, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1),kernel_initializer="he_normal", name=conv_name_base, use_bias=False, kernel_regularizer=regularizer)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, regularizer, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, regularizer, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter