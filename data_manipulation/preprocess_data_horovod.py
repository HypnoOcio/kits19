import numpy as np
import os
from numpy import load
from generator_train import cut_off_background

def load_train_data(iter_lst):
    '''func loads whole dataset'''
        vol,seg = None,None

        for i in iter_lst:
                start = time.time()
                dict_data = load( './kits_cases/big_vol_{num}.npz'.format(num = i) )
                # extract the first array
                tmp_vol = dict_data['arr_0']
                dict_data = load('./kits_cases/big_seg_{num}.npz'.format(num = i))
                # extract the first array
                tmp_seg = dict_data['arr_0']
                if vol is None or seg is None:
                        vol,seg = tmp_vol,tmp_seg
                else:
                        vol = np.concatenate((vol, tmp_vol), axis=0)
                        seg = np.concatenate((seg, tmp_seg), axis=0)
                end = time.time()
                print(f'file_{i} took {end-start} sec')
        # cubic/bilinear/nearest interpolation is used while creating images so values are not strictly 0.0 or 1.0
        # values need to be rounded [0.0 ,1.0]
        seg[seg > 0.20] = 1.0
        seg[seg < 0.20] = 0.0
        return vol,seg

def load_train_data_file_by_file(iter_list):
    '''func loads patients in iter_list'''
        vol,seg = [],[]

        for i in iter_list:
                if os.path.exists( '/storage/brno8/home/detkotom/bakalarka/separated_files/big_vol_small_{num}.npz'.format(num = i) ) == False:
                        continue
                dict_data = load( '/storage/brno8/home/detkotom/bakalarka/separated_files/big_vol_small_{num}.npz'.format(num = i) )
                # extract the first array
                tmp_vol = dict_data['arr_0']
                dict_data = load('/storage/brno8/home/detkotom/bakalarka/separated_files/big_seg_small_{num}.npz'.format(num = i))
                # extract the first array
                tmp_seg = dict_data['arr_0']
                vol.append(tmp_vol)
                seg.append(tmp_seg)
        vol = np.concatenate(vol, axis = 0)
        seg = np.concatenate(seg, axis = 0)
        # cubic/bilinear/nearest interpolation is used while creating images so values are not strictly 0.0 or 1.0
        # values need to be rounded [0.0 ,1.0]
        seg[seg > 0.20] = 1.0
        seg[seg < 0.20] = 0.0
        return vol,seg

def gen_overlap_patches(x,y,patch_size,overlap):
    '''func generates overlapping patches for 3D network'''
        if overlap >= patch_size or len(x) < patch_size:
                raise ValueError("overlap has to be smaller then patch_size or len(x) bigger then patch_size")
        start, end = 0, patch_size
        data_x,data_y = [],[]
        while True:
                if end > len(x):
                        end = len(x)
                        start = end - patch_size

                data_x.append(x[start:end])
                data_y.append(y[start:end])
                if end == len(x):
                        break

                start = end   - overlap
                end   = start + patch_size

        x_new = np.concatenate(data_x,axis=0)
        y_new = np.concatenate(data_y,axis=0)
        return x_new,y_new


def reshape_to_right_patch_size(x_train, y_train, patch_size):
    '''func reshapes patches so they are identical'''
        #width and height are the same
        height = x_train.shape[-3]
        width  = x_train.shape[-2]
        rest = len(x_train) % patch_size
        if( rest != 0 ):
                x_train = x_train[:-rest]
                y_train = y_train[:-rest]
        #reshape patch x width x height x 3
        x_train_reshaped = x_train.reshape( ( int ( len(x_train) / patch_size ), patch_size, height, width, 3) )
        y_train_reshaped = y_train.reshape( ( int ( len(x_train) / patch_size ), patch_size, height, width, 3) )
        return x_train_reshaped, y_train_reshaped


def rgb_to_onehot(rgb_arr, color_dict):
    '''func creates from RGB a one-hot representation of segmentation mask '''
        num_classes = len(color_dict)
        shape = rgb_arr.shape[:2]+(num_classes,)
        arr = np.zeros( shape, dtype=np.int8 )
        for i, cls in enumerate(color_dict):
                arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
        return arr

def onehot_to_rgb(onehot, color_dict):
    '''func creates from one-hot representation a RGB segmentation mask '''
        single_layer = np.argmax(onehot, axis=-1)
        output = np.zeros( onehot.shape[:2]+(3,) )
        for k in color_dict.keys():
                output[single_layer==k] = color_dict[k]
        return np.uint8(output)

def prepare_data_for_softmax(y_train_patient, num_classes):
        color_dict = {0: (255,   0, 0),
              1: (0, 0,   0),
              2: (0, 0,   255)}
        if len(color_dict) != num_classes:
                raise ValueError('Color_dict len() has to be same as num_classes.')

        softmax_out_patient = np.zeros( y_train_patient.shape[:-1] + ( num_classes, ) )
        for i, pic in enumerate(y_train_patient):
                tmp_img = 255 * pic
                tmp_img = tmp_img.astype(np.uint8)

                one_hot = rgb_to_onehot(tmp_img,color_dict)
                softmax_out_patient[i] = one_hot

        return softmax_out_patient


def get_data(iter_list,model_2D,cut_off,want_overlap,patch_size,overlap,use_softmax,num_classes,multiply_tumors):
        x,y = [],[]
        for patient in iter_list:
                if os.path.exists( '/storage/brno8/home/detkotom/bakalarka/separated_files/big_vol_small_{num}.npz'.format(num = patient) ) == False:
                        continue
                x_train_patient, y_train_patient = load_train_data_file_by_file( [patient] )
                if cut_off == True: #cutting of background
                        x_train_patient, y_train_patient = cut_off_background( x_train_patient, y_train_patient, to_cut = 0 )
                        #create overlap for 3D model
                if multiply_tumors > 1: #cutting of tumors
                        multiply_tumors = int(multiply_tumors)
                        x_train_tumors, y_train_tumors = cut_off_background( x_train_patient, y_train_patient, to_cut = 1 )

                        multiplied_x = np.tile(x_train_tumors,(multiply_tumors,1,1,1))
                        multiplied_y = np.tile(y_train_tumors,(multiply_tumors,1,1,1))

                        x_train_patient = np.concatenate((x_train_patient,multiplied_x))
                        y_train_patient = np.concatenate((y_train_patient,multiplied_y))
                if use_softmax == True:
                        y_train_patient = prepare_data_for_softmax(y_train_patient, num_classes)
                if model_2D == False and want_overlap == True:
                        #if patch_size = 32 then we lose 4 patients (100images) - bearable losses
                        if len(x_train_patient) < patch_size:
                                continue
                        x_train_patient, y_train_patient = gen_overlap_patches(x_train_patient, y_train_patient, patch_size, overlap)
                #create right shape of training data
                if model_2D == False:
                        x_train_patient, y_train_patient = reshape_to_right_patch_size(x_train_patient, y_train_patient, patch_size)
                x.append(x_train_patient)
                y.append(y_train_patient)

        x_train = np.concatenate(x, axis=0)
        y_train = np.concatenate(y, axis=0)
        return x_train, y_train
