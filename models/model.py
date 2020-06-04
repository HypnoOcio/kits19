from tensorflow import keras

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
from keras import regularizers

#import keras.backend as K
from keras import backend as K
K.set_image_data_format('channels_last')

from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose

from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization

# best approaches for winning segmentation https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741

def Unet2D_small(input_size,use_softmax, num_classes,regularizer = None):

    if use_softmax == True:
        activation = "softmax"
    else:
        activation = "sigmoid" 

    inputs = Input(input_size)
    #background,kidney,tumor
    num_classes = num_classes 
    #in 128x128x3
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(inputs)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    drop4 = conv4 #Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)


    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    drop5 = conv5

    up6   = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6   = BatchNormalization(axis=-1)(up6)
    merge6= concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization(axis=-1)(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Conv2D(10, 3, activation = 'relu', padding = 'same', kernel_regularizer = regularizer, kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7) 
    conv7 = Conv2D(num_classes, (1,1), activation = activation)(conv7)

    model = Model(input = inputs, output = conv7)
    
    return model
    
def Unet2D_big(input_size):
    #in 512x512x3
    #do not use InputLayer - keras version 2.2.x has bug there - can not load such weights after checkpoint of model
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = conv4
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = conv5 

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    return model

def Unet3D(input_shape, use_softmax, n_labels = 3, n_filters=32, depth=4, regularizer = None ):
    if use_softmax == True:
        activation = "softmax"
    else:
        activation = "sigmoid"      

    inputs = Input(input_shape)
    cnn_chain = inputs
    contracting_convs = []

    # Encoder part
    for i in range(0, depth):
        neurons = n_filters * 2**i
        cnn_chain, last_conv = contracting_layer(cnn_chain, neurons, regularizer)
        contracting_convs.append(last_conv)

    # Middle part
    neurons = n_filters * 2**depth
    cnn_chain = middle_layer(cnn_chain, neurons, regularizer)

    # Decoder part
    for i in reversed(range(0, depth)):
        neurons = n_filters * 2**i
        cnn_chain = expanding_layer(cnn_chain, neurons, contracting_convs[i], regularizer)

    conv_out = Conv3D(n_labels, (1, 1, 1), activation=activation)(cnn_chain)
    model = Model(inputs=[inputs], outputs=[conv_out])
    return model

def contracting_layer(input, neurons, regularizer):
    input = BatchNormalization(axis=-1)(input)
    conv1 = Conv3D(neurons, (3,3,3), activation='relu', kernel_regularizer = regularizer, padding='same')(input)

    conv1 = BatchNormalization(axis=-1)(conv1)
    conv2 = Conv3D(neurons, (3,3,3), activation='relu', kernel_regularizer = regularizer, padding='same')(conv1)

    conv2 = BatchNormalization(axis=-1)(conv2)
    conc1 = concatenate([input, conv2], axis=4)

    pool = MaxPooling3D(pool_size=(2, 2, 2))(conc1)
    return pool, conv2

def middle_layer(input, neurons, regularizer):
    input = BatchNormalization(axis=-1)(input)
    conv_m1 = Conv3D(neurons, (3, 3, 3), activation='relu', kernel_regularizer = regularizer, padding='same')(input)

    conv_m1 = BatchNormalization(axis=-1)(conv_m1)
    conv_m2 = Conv3D(neurons, (3, 3, 3), activation='relu', kernel_regularizer = regularizer, padding='same')(conv_m1)

    conc1 = concatenate([input, conv_m2], axis=4)
    return conc1

def expanding_layer(input, neurons, concatenate_link, regularizer):
    up = concatenate([Conv3DTranspose(neurons, (2, 2, 2), strides=(2, 2, 2),
                    padding='same')(input), concatenate_link], axis=4)
    
    up = BatchNormalization(axis=-1)(up)
    conv1 = Conv3D(neurons, (3, 3, 3), activation='relu', kernel_regularizer = regularizer, padding='same')(up)

    conv1 = BatchNormalization(axis=-1)(conv1)
    conv2 = Conv3D(neurons, (3, 3, 3), activation='relu', kernel_regularizer = regularizer, padding='same')(conv1)
    
    conc1 = concatenate([up, conv2], axis=4)

    return conc1
