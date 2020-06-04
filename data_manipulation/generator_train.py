
import numpy as np
from data_load import get_images_train, load_data
from constants import *
import random
from collections import defaultdict
import cv2
from keras.preprocessing.image import Iterator, ImageDataGenerator

#source for advanced distributed data generator - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator():
# Generator is used only for reading data frmo disk. It is not used for distributed training.'
    def __init__(self, list_IDs,batch_size=4, dim=(512,512), rescale = 512, n_channels=3, shuffle=True):

        self.list_of_files = list(list_IDs)
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.rescale = rescale
        #files have different lengths so we need to know what files are finished
        self.already_read = {}
        #dict so I do not need to always reload the whole file from disk
        self.loaded_files = {}
        self.already_processed_files = 0
        self.refresh_for_epoch()
    
    def __iter__(self):
        return self

    def __next__(self):
        x , y = self.get_batch() 
        return x, y

    def refresh_for_epoch(self):
        'Shuffle and restore everything for each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.list_of_files)

        self.already_processed_files = 0
        #we have loaded nothing here
        self.loaded_files = {}   
        #we have read 0 from each file so far
        self.already_read = { i:0 for i in self.list_of_files }

    def get_batch(self):
        'Generate one batch of data'
        
        if self.already_processed_files >= len(self.list_of_files):
            self.refresh_for_epoch()
        num_of_file = self.list_of_files[ self.already_processed_files ]   

        finished, vol_ims, seg_ims, _ = self.get_photos_from_file( num_of_file )
        if finished == True:
            self.already_processed_files += 1
            self.loaded_files.pop( num_of_file )

        #X,Y    
        return vol_ims, seg_ims

    def rescale_3D_images(self, images_3D):
        #get number of 3D images in stack
        img_stack = images_3D.shape[0]

        width, height = self.rescale, self.rescale
        img_stack_sm = np.zeros( (img_stack, width, height, 3) )

        for idx in range(img_stack):
            img = images_3D[idx]
            img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            img_stack_sm[idx] = img_sm

        return img_stack_sm

    def get_photos_from_file(self,num_of_file):
        '''func returns a batch of photos for specific file a indicator whether file ends or not'''
        keys = self.loaded_files.keys()

        # photos generator has already seen couple of pictures from this file 
        already_seen_pics = self.already_read[num_of_file]

        if num_of_file not in keys:
            vol, seg = load_data(DATA_PATH, num_of_file)
            # each loaded file contains 3 files - vol,seg,viz
            self.loaded_files[ num_of_file ] = get_images_train(vol, seg, startIdx = None, endIdx = None)
            # rescale if demanded size is not default size
            if self.rescale != 512:
                self.loaded_files[ num_of_file ][0] = self.rescale_3D_images(self.loaded_files[ num_of_file ][0])
                self.loaded_files[ num_of_file ][1] = self.rescale_3D_images(self.loaded_files[ num_of_file ][1])
                self.loaded_files[ num_of_file ][2] = self.rescale_3D_images(self.loaded_files[ num_of_file ][2])

        vol_ims = self.loaded_files[ num_of_file ][0][ already_seen_pics : already_seen_pics + self.batch_size ]
        seg_ims = self.loaded_files[ num_of_file ][1][ already_seen_pics : already_seen_pics + self.batch_size ]
        viz_ims = self.loaded_files[ num_of_file ][2][ already_seen_pics : already_seen_pics + self.batch_size ]
        vol_ims = vol_ims.astype(np.float32)
        vol_ims /= 255.
        seg_ims = seg_ims.astype(np.float32)
        seg_ims /= 255.
        viz_ims = viz_ims.astype(np.float32)
        viz_ims /= 255.
        self.already_read[ num_of_file ] += self.batch_size

        file_finished = False

        if self.already_read[ num_of_file ] >= self.loaded_files[ num_of_file ][0].shape[0]:
            file_finished = True
        return (file_finished ,vol_ims, seg_ims, viz_ims)

class MyIterator(Iterator):
  """This is a toy example of a wrapper around ImageDataGenerator"""

  def __init__(self, n, batch_size, shuffle, seed, training, x, y):
    super().__init__(n, batch_size, shuffle, seed)

    self.num_of_images = n
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.seed = seed
    self.training = training
    self.x_imgs = x
    self.y_imgs = y
    self.indices = np.arange(x.shape[0])

    self.generator = ImageDataGenerator(
                                #rotation_range=5,
                                #zoom_range=0.05,
                                width_shift_range=[-10,10],
                                height_shift_range =[-10,10],
                                horizontal_flip = True)

  def on_epoch_end(self):
    if self.shuffle == True:
        np.random.shuffle(self.indices)

  def __len__(self):
    return int( len(self.x_imgs) // self.batch_size )

  def __iter__(self):
    return self

  def _get_batches_of_transformed_samples(self, index_array):
    """Gets a batch of transformed samples from array of indices"""

    # Get a batch of image data
    list_ids = self.indices[index_array]
    
    batch_x = self.x_imgs[list_ids].copy()
    batch_y = self.y_imgs[list_ids].copy()

    if self.training == True:
      transform_params = self.generator.get_random_transform(batch_x[0].shape)
      for i, (x, y) in enumerate(zip(batch_x, batch_y)):
        batch_x[i] = self.generator.apply_transform(x, transform_params)
        batch_y[i] = self.generator.apply_transform(y, transform_params)

    return batch_x, batch_y

class MyIterator3D(Iterator):
  """This is a toy example of a wrapper around ImageDataGenerator"""

  def __init__(self, n, batch_size, shuffle, seed , training , x, y):
    super().__init__(n, batch_size, shuffle, seed)

    self.num_of_images = n
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.seed = seed
    self.training = training
    self.x_imgs = x
    self.y_imgs = y
    self.indices = np.arange(x.shape[0])

    self.generator = ImageDataGenerator(
                                #rotation_range=5,
                                #zoom_range=0.05,
                                width_shift_range=[-10,10],
                                height_shift_range =[-10,10],
                                horizontal_flip = True)
      
  def on_epoch_end(self):
     if self.shuffle == True:
         np.random.shuffle(self.indices)

  def __len__(self):
    return int( len(self.x_imgs) // self.batch_size )

  def __iter__(self):
    return self
     
#  def __getitem__(self, index_array):
  def _get_batches_of_transformed_samples(self, index_array):
    """Gets a batch of transformed samples from array of indices"""

    # Get a batch of image data
    list_ids = self.indices[index_array]
    
    batch_x = self.x_imgs[list_ids].copy()
    batch_y = self.y_imgs[list_ids].copy()

    if self.training == True:
      #get transformation for one image of patch of batch
      transform_params = self.generator.get_random_transform(batch_x[0][0].shape)
      #iterate over all patches of batch
      for j, (patch_x,patch_y) in enumerate(zip(batch_x, batch_y)): 
        #iterate over all images in patch
        for i,(x,y) in enumerate(zip(patch_x, patch_y)):
          patch_x[i] = self.generator.apply_transform(x, transform_params)
          patch_y[i] = self.generator.apply_transform(y, transform_params)   

    return batch_x, batch_y

#if to_cut == 0 then cut off background
#if to_cut == 1 then cut of everything expect of tumors
def cut_off_background(x_train,y_train,to_cut):
    """func used for balancing dataset"""
    data_x, data_y = x_train, y_train
    lst_x = []
    lst_y = []
    if to_cut == 0:
        for i in range(0,len(data_y) ):
            if np.amax(data_y[i]) > 0.5:
                lst_x.append(data_x[i])
                lst_y.append(data_y[i])
    else:
        #128x128x3 is dimmension of y_true - here filtering blue color (tumors)
        mask_zeros = np.zeros((128,128,1))
        mask_ones = np.ones((128,128,1))
        filt = np.concatenate((mask_zeros,mask_zeros,mask_ones), axis=2)
        filt = np.squeeze(filt)
        for i in range(len(data_y)):
            tmp = np.multiply(data_y[i],filt)
            if np.max(tmp) > 0.5:
                lst_x.append(data_x[i])
                lst_y.append(data_y[i])

    data_y = np.stack([lst_y])
    data_y = np.squeeze(data_y, axis=0)
    data_x = np.stack([lst_x])
    data_x = np.squeeze(data_x, axis=0)

    return data_x,data_y

