from pspNet_small import pspnet_small
from pspNet_big import pspnet_big
from denseUNet_small import Dense_UNet_small
from denseUNet_big import Dense_UNet_big
from deeplabv3plus import Deeplabv3
from model import  Unet3D, Unet2D_small, Unet2D_big
from keras.layers import Input

def get_model_name(num):
        models = { 0 : ("Unet2D_small",True), 1 : ("Unet3D",False),\
                   2 : ("pspnet_small",True), 3 : ("pspnet_big",True),\
                   4 : ("Dense_UNet_small",True), 5 : ("Dense_UNet_big",True),\
                   6 : ("Deeplabv3",True) }
        if num > 6 or num < 0:
                raise ValueError(f'Model with num {num} is not implemented yet.')
        else:
                return models[num]


def get_model_architecture(model_name, picture_size, num_classes, patch_size, regularizer, load_weights_from_internet = False ):
''' fucn returns creates an architecture of a model based on given model_name'''

        if model_name == 'Unet2D_small':
                model = Unet2D_small(input_size = (picture_size, picture_size, 3), use_softmax = True, num_classes = num_classes, regularizer=regularizer )

        elif model_name == 'pspnet_small':
                model = pspnet_small(num_classes = num_classes, input_shape = (picture_size, picture_size, 3), regularizer = regularizer )

        elif model_name == 'pspnet_big':
                model = pspnet_big(num_classes = num_classes, input_shape = (picture_size, picture_size, 3), regularizer = regularizer)

        elif model_name == 'Dense_UNet_small':
                input_shape = (picture_size,picture_size,3)
                input_img = Input(input_shape, name='model_input')
                model = Dense_UNet_small(input_img,num_classes = num_classes, dropout=0.05, regularizer = regularizer)

        elif model_name == 'Dense_UNet_big':
                model = Dense_UNet_big(input_shape = (picture_size, picture_size, 3), num_classes = num_classes, regularizer = regularizer)

        elif model_name == 'Deeplabv3':
                model = Deeplabv3( input_shape = (picture_size,picture_size,3), num_classes=num_classes, regularizer = regularizer, load_weights_from_internet = load_weights_from_internet)

        elif model_name == "Unet3D":
                model = Unet3D(input_shape = ( patch_size, picture_size, picture_size, 3 ), n_filters=48 ,depth = 3, use_softmax = True, n_labels = num_classes)
        else:
                print("No model with such name. Exit...")
                exit()
        return model
