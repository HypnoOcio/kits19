import argparse
import scipy.misc
import nibabel as nib
import numpy as np
import io
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline 
#%config InlineBackend.figure_format = 'retina'
from pathlib import Path

from constants import *


def load_case(cid):
    vol = nib.load('test.nii.gz')
    seg = None #nib.load(str(case_path / "segmentation.nii.gz"))
    return vol, seg


"plt_show(out/255) if want to show output of this func"
def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


def class_to_color(segmentation, k_color, t_color):
    # initialize output to zeros
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    # set output to appropriate color at each location
    seg_color[np.equal(segmentation,1)] = k_color
    seg_color[np.equal(segmentation,2)] = t_color
    return seg_color


def overlay(volume_ims, segmentation_ims, segmentation, alpha):
    # Get binary array for places where an ROI lives
    segbin = np.greater(segmentation, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    overlayed = np.where(
        repeated_segbin,
        np.round(alpha*segmentation_ims+(1-alpha)*volume_ims).astype(np.uint8),
        np.round(volume_ims).astype(np.uint8)
    )
    return overlayed


def visualize(cid, destination, hu_min=DEFAULT_HU_MIN, hu_max=DEFAULT_HU_MAX, 
    k_color=DEFAULT_KIDNEY_COLOR, t_color=DEFAULT_TUMOR_COLOR,
    alpha=DEFAULT_OVERLAY_ALPHA):
    # Prepare output location
    out_path = Path(destination)
    if not out_path.exists():
        out_path.mkdir()  

    # Load segmentation and volume
    vol, seg = load_case(cid)
    vol = vol.get_data()
    # seg = seg.get_data()
    # seg = seg.astype(np.int32)
    
    # Convert to a visual format
    vol_ims = hu_to_grayscale(vol, hu_min, hu_max)
    # seg_ims = class_to_color(seg, k_color, t_color)
    
    # Overlay the segmentation colors
    # viz_ims = overlay(vol_ims, seg_ims, seg, alpha)

    # Save individual images to disk
    for i in range(vol_ims.shape[0]):
        fpath = out_path / ("{:05d}.png".format(i))
        scipy.misc.imsave(str(fpath), vol_ims[i])