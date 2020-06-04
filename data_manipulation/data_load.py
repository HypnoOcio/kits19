import os
import numpy as np
import nibabel as nib
from Kits19_methods import hu_to_grayscale,class_to_color,overlay
from constants import *


def get_images_train(vol,seg,startIdx,endIdx):
        """func returns vol_ims_small - it needs to be scaled down if printed plt.show(vol_ims_small[5]/255)
                        seg_ims_small 
                        viz_ims_small """

        if startIdx == None or endIdx == None:
            vol_ims_small = hu_to_grayscale(vol, DEFAULT_HU_MIN, DEFAULT_HU_MAX)
            seg_ims_small = class_to_color(seg, DEFAULT_KIDNEY_COLOR, DEFAULT_TUMOR_COLOR)
            viz_ims_small = overlay(vol_ims_small, seg_ims_small, seg, DEFAULT_OVERLAY_ALPHA)       

        else:
            vol_ims_small = hu_to_grayscale(vol[ startIdx : endIdx ], DEFAULT_HU_MIN, DEFAULT_HU_MAX)
            seg_ims_small = class_to_color(seg[ startIdx : endIdx ], DEFAULT_KIDNEY_COLOR, DEFAULT_TUMOR_COLOR)
            viz_ims_small = overlay(vol_ims_small, seg_ims_small, seg[ startIdx : endIdx ], DEFAULT_OVERLAY_ALPHA)
        
        return [vol_ims_small,seg_ims_small,viz_ims_small]


def get_file_name(num_of_file):
        """func from serial number creates name of file"""
        name = ''
        str_i = str(num_of_file)
        if len(str_i) == 1:
            name = "case_0000" + str_i
        elif len(str_i) == 2:
            name = "case_000" + str_i
        else:
            name = "case_00" + str_i       
        return name


def load_data(data_path,num_of_file):
        """func used to compact loading data from files - used only for training data
        # datapath - path to directory
        # num_of_file - serial number of file --- in our case 0 - 209"""
        # example load
        imaging_name = 'imaging.nii.gz'
        segmentation_name = 'segmentation.nii.gz'
        
        file_name = get_file_name(num_of_file)
        case = nib.load(os.path.join(data_path, file_name, imaging_name))
        seg = nib.load(os.path.join(data_path, file_name, segmentation_name))
        
        vol = case.get_data()
        seg = seg.get_data()
        seg = seg.astype(np.int32)
        
        return (vol, seg)