
import sys
import os
google_colab = False

if 'google.colab' in sys.modules:
  google_colab = True  

# Constants
DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_TUMOR_COLOR = [0, 0, 255]
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3

IMAGING_NAME = 'imaging.nii.gz'
SEGMENTATION_NAME = 'segmentation.nii.gz'

DATA_PATH = ''
if google_colab:
    DATA_PATH = 'drive/My Drive/kits/kits_cases_subset'
else:
    DATA_PATH = os.environ['SCRATCHDIR'] + '/kits_cases_subset'

