#controlled randomness
#https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras/52897216#52897216
#https://github.com/horovod/horovod/issues/173

#easy to have controlled randomness one one CPU - harder to set it over mutliple GPUs on multiple machines
import horovod.keras as hvd
# Horovod: initialize Horovod.
hvd.init()

import sys

seed_value = 42

import os
os.environ['PYTHONHASHSEED']=str(seed_value + hvd.rank() )

import random
random.seed(seed_value + hvd.rank() )


import numpy as np
np.random.seed(seed_value + hvd.rank())

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value + hvd.rank() )

sys.path.append('./data_manipulation')
sys.path.append('./models')

import shutil
import json

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential,load_model
from keras.regularizers import l1, l2
from keras import backend as K
import math
import tensorflow as tf
import numpy as np
from numpy import load

from constants import *
import Kits19_methods as k19
import data_load as dl
import generator_train as gt
from store_learning import create_images_of_training
from class_weight import get_class_weights

from model import  Unet3D, Unet2D_small, Unet2D_big
from pspNet_small import pspnet_small
from pspNet_big import pspnet_big
from denseUNet_small import Dense_UNet_small
from denseUNet_big import Dense_UNet_big
from deeplabv3plus import Deeplabv3

from loss_function import make_loss, softmax_dice_loss
from generator_train import MyIterator, MyIterator3D, cut_off_background #cut_off_background removes blank images only with black pixels
from preprocess_data_horovod import get_data
from load_model import get_model_name, get_model_architecture
import time

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

print(f'num of GPUS = {hvd.size()} ')

interactive = True
if interactive:
    PATH = os.getcwd()
else:
    PATH = os.environ['SCRATCHDIR']

sys.path.append( PATH +"/kits19_tumor" )

log_dir            = "/logs"
model_dir          = "/model_weights"
timestr            = time.strftime("%Y_%m_%d-%H_%M_%S")
all_model_here     = "/outputs"

alpha              = 0.5
beta               = 0.5
learning_rate      = 0.0001 
epochs             = 5 
use_class_weight  = True
use_regularizer   = True 

resume_from_epoch  = 0
warmup_epochs     = 3 
period_logging    = 6 
multiply_tumors   = 4 #how much balance dataset

picture_size      = 128
use_softmax       = True
num_classes       = 3

regularizer       = l2(0.0001) if use_regularizer == True else None
#setup for 3D Unet
patch_size = 32
overlap    = 12

#0 - Unet2D_small
#1 - Unet3D
#2 - pspnet_small
#3 - pspnet_big
#4 - Dense_UNet_small
#5 - Dense_UNet_big
#6 - Deeplabv3+
model_num = 0
model_name, model_2D = get_model_name(model_num)


model_store_here = model_name+"_"+timestr
dir_model = PATH + os.path.join( all_model_here, model_store_here)

if hvd.rank() == 0:
        if os.path.isdir( os.path.join( PATH, all_model_here ) == False:
                print("Dir for models does not exist.")
                try:
                        os.mkdir( os.path.join( PATH, all_model_here) , mode = 0o770 )
                except:
                        pass
        if os.path.isdir( os.path.join( PATH, all_model_here, model_store_here) ) == False:
                print("Dir for specific model does not exist.")
                try:
                        os.mkdir( dir_model, mode = 0o770 )
                except:
                        pass

if model_2D == True:
        cut_off = True
        want_overlap = False
else:
        cut_off = True
        want_overlap = True

# If set > 0, will resume training from a given checkpoint.
# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

if model_2D == False:
    model_weight_format = '/weights-3D-{epoch}.h5'
    last_weights = '/weights_3D.h5'
else:
    model_weight_format = '/weights-2D-{epoch}.h5'
    last_weights = '/weights_2D.h5'

iter_list_train = list(range(0,120)) + list(range(140,200)) #list(range(0,2))  #list(range(0,210))
iter_list_valid = list(range(120,140))
iter_list_test = list(range(200,210))

start_time = time.time()

x_train,y_train = get_data(iter_list = iter_list_train, model_2D = model_2D, cut_off=cut_off,
                                 want_overlap = want_overlap, patch_size = patch_size, overlap = overlap,
                                 use_softmax = use_softmax, num_classes = num_classes,
                                 multiply_tumors = multiply_tumors)

x_val, y_val    = get_data(iter_list = iter_list_valid, model_2D = model_2D, cut_off = False,
                                 want_overlap = False, patch_size = patch_size, overlap = overlap,
                                 use_softmax = use_softmax, num_classes = num_classes,
                                 multiply_tumors = 1)

x_test, y_test    = get_data(iter_list = iter_list_test, model_2D = model_2D, cut_off = False,
                                 want_overlap = False, patch_size = patch_size, overlap = overlap,
                                 use_softmax = use_softmax, num_classes = num_classes,
                                 multiply_tumors = 1)


end_time = time.time()


print(f'time for patient {end_time-start_time}')

cls_weight = None
if use_class_weight == True:
        print("Weight classes.")
        cls_weight = get_class_weights(y_train)


num_GPUS          = hvd.size()
num_of_scans      = x_train.shape[0]
batch_size        = 64 if model_2D == True and model_name != 'Dense_UNet_big'  else 4
batches_per_epoch = math.ceil( num_of_scans / batch_size )
val_batches       = math.ceil (len(x_val)   / batch_size )

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

model = get_model_architecture(model_name, picture_size, num_classes, patch_size, regularizer )
if os.path.exists( dir_model + last_weights ) and model_2D == True:
        if hvd.rank() == 0:
                for i in range(10):
                        print("loading weights 2D ")
        model.load_weights( os.path.join( dir_model, last_weights ) )
elif os.path.exists( dir_model + last_weights ) and model_2D == False:
        if hvd.rank() == 0:
                for i in range(10):
                        print("loading weights 3D ")
        model.load_weights( dir_model + last_weights )

opt = keras.optimizers.Adam(learning_rate = learning_rate)
# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)
metrics = [make_loss('softmax_dice_loss', alpha = 0.5, beta = 0.5, class_weights = None),
           make_loss('tversky_loss', alpha = 0.5, beta = 0.5, class_weights = cls_weight),
           make_loss('tversky_loss', alpha = 1.0, beta = 1.0, class_weights = None),
           make_loss('tversky_loss', alpha = 1.0, beta = 1.0, class_weights = cls_weight),
           make_loss('tversky', alpha = 0.5, beta = 0.5, class_weights = None),
           make_loss('focal_loss', alpha = 0.25, beta = 2.0, class_weights = None),
           make_loss('weighted_cross_entropy', alpha = 1.0, beta = 3.0, class_weights = None),
           make_loss('balanced_cross_entropy', alpha = 1.0, beta = 0.2, class_weights = None)]


loss_name = "tversky_loss"
loss = make_loss(loss_name, alpha = alpha, beta = beta, class_weights = cls_weight)

model.compile( loss = loss,
               optimizer = opt, metrics = metrics )

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1 if hvd.rank() == 0 else 1 ),

    keras.callbacks.ReduceLROnPlateau(patience = 3, factor = 0.95, verbose = 0),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=150, multiplier=0.6),
]

# Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(filepath =  dir_model + model_weight_format, save_weights_only = True, period = period_logging))
    #callbacks.append(keras.callbacks.ModelCheckpoint(filepath =  dir_model + model_weight_format, save_weights_only = True, monitor='val_jacc_w_loss',mode='min' , save_best_only = True,))
    callbacks.append(keras.callbacks.TensorBoard( dir_model ) )

if model_2D == True:
    train_gen = MyIterator(   len(x_train), batch_size = batch_size, shuffle = True, seed = 42, training = True , x = x_train, y = y_train)
    valid_gen = MyIterator(   len(x_val)  , batch_size = batch_size, shuffle = True, seed = 42, training = False, x = x_val, y = y_val)
    test_gen  = MyIterator(   len(x_test) , batch_size = batch_size, shuffle = False, seed = 42, training = False , x = x_test, y = y_test)
else:
    train_gen = MyIterator3D( len(x_train), batch_size = batch_size, shuffle = True, seed = 42, training = True,  x = x_train, y = y_train)
    valid_gen = MyIterator3D( len(x_val)  , batch_size = batch_size, shuffle = True, seed = 42, training = False, x = x_val, y = y_val)
    test_gen  = MyIterator3D( len(x_test) , batch_size = batch_size, shuffle = False, seed = 42, training = False, x = x_test, y = y_test)

start = time.time()

history = model.fit_generator(train_gen,
          steps_per_epoch  = int(batches_per_epoch / num_GPUS),
          callbacks        = callbacks,
          epochs           = epochs,
          workers          = num_GPUS,
          initial_epoch    = resume_from_epoch,
          class_weight     = cls_weight,
          verbose          = 1 if hvd.rank() == 0 else 0 ,
          validation_data  = valid_gen,
          validation_steps = val_batches // hvd.size())

end = time.time()


if hvd.rank() == 0:
        path_to_save = dir_model
        create_images_of_training(history,path_to_save)

#show times only for first worker
if hvd.rank() == 0:
        print(f'first training took {end-start} sec w/ {hvd.local_rank()} GPU')
        print(f'num of GPUS = {hvd.size()} ')
        print(f'learning rate of model is : {model.optimizer.get_config()["learning_rate"]}')

if hvd.rank() == 0:
        print('\n# Evaluate')
        result = model.evaluate(test_gen)
        print(dict(zip(model.metrics_names, result)))


if hvd.rank() == 0:
        model.save_weights( dir_model + last_weights )
        shutil.copyfile(PATH + "/out.txt", dir_model + "/out.txt")
        shutil.copyfile(PATH + "/err.txt", dir_model + "/err.txt")
        #shutil.copyfile(PATH + "/model.py", dir_model + "/model.py")
        #shutil.copyfile(PATH + "/work_here_horovod.py", dir_model + "/work_here_horovod.py")



        out_dict = {}
        out_dict['learning_rate']       = learning_rate
        out_dict['epochs']              = epochs
        out_dict['model_store_here']    = model_store_here
        out_dict['loss_name']           = loss_name
        out_dict['alpha']               = alpha
        out_dict['beta']                = beta
        out_dict['multiply_tumors']     = multiply_tumors
        out_dict['resume_from_epoch']   = resume_from_epoch
        out_dict['model_name']          = model_name
        out_dict['model_2D']            = model_2D
        out_dict['warmup_epochs']       = warmup_epochs
        out_dict['period_logging']      = period_logging
        out_dict['picture_size']        = picture_size
        out_dict['use_softmax']         = use_softmax
        out_dict['use_class_weight']    = use_class_weight
        out_dict['use_regularizer']     = use_regularizer
        out_dict['num_classes']         = num_classes
        out_dict['cut_off']             = cut_off
        out_dict['want_overlap']        = want_overlap
        out_dict['patch_size']          = patch_size
        out_dict['overlap']             = overlap
        out_dict['num_of_scans']        = num_of_scans
        out_dict['batch_size']          = batch_size
        out_dict['batches_per_epoch']   = batches_per_epoch
        out_dict['val_batches_per_epoch']= val_batches
        with open(dir_model+"/basic_info.txt", 'w') as f:
                for key, value in out_dict.items():
                        f.write(key + " : " + str(value) + "\n")