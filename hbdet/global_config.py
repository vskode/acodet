##############################################################################

# THIS FILE IS ONLY MEANT TO BE EDITED IF YOU ARE SURE!

# PROGRAM FAILURE IS LIKELY TO OCCUR IF YOU ARE UNSURE OF THE
# CONSEQUENCES OF YOUR CHANGES.

# IF YOU HAVE CHANGED VALUES AND ARE ENCOUNTERING ERRORS, 
# STASH YOUR CHANGES ('git stash' in a git bash console in hbdet directory)

##############################################################################

import yaml

with open('user_config.yml', 'r') as f:
    config = yaml.safe_load(f)
    
## Preprocessing Parameters
STFT_FRAME_LEN = 1024
N_FREQ_BINS = 128
FMIN = 50
FMAX = 1000
SR = 2000
CONTEXT_WIN = 7755

# downsample every audio file to this frame rate to ensure comparability
DOWNSAMPLE_SR = 2000

## Settings for Creation of Tfrecord Dataset
# limit of context windows in a tfrecords file
TFRECS_LIM = 600
# train/test split
TRAIN_RATIO = 0.7
# test/val split
TEST_VAL_RATIO = 0.7

## Model Parameters
# threshold for predictions
THRESH = config['thresh']
# prediction window limit
PRED_WIN_LIM = 50

# calculated global variables
FFT_HOP = (CONTEXT_WIN - STFT_FRAME_LEN) // (N_FREQ_BINS - 1)
PRED_BATCH_SIZE = PRED_WIN_LIM * CONTEXT_WIN   
           
## Paths
# dataset destination directory - save tfrecord dataset to this directory (leave unchanged )
TFREC_DESTINATION = '../Data/Datasets'
ANNOTATION_DESTINATION = config['annotation_destination']
ANNOTATION_SOURCE = config['annotation_source']
SOUND_FILES_SOURCE = config['sound_files_source']
