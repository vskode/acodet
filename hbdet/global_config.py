##############################################################################

# THIS FILE IS ONLY MEANT TO BE EDITED IF YOU ARE SURE!

# PROGRAM FAILURE IS LIKELY TO OCCUR IF YOU ARE UNSURE OF THE
# CONSEQUENCES OF YOUR CHANGES.

# IF YOU HAVE CHANGED VALUES AND ARE ENCOUNTERING ERRORS, 
# STASH YOUR CHANGES ('git stash' in a git bash console in hbdet directory)
# AND THEN PULL AGAIN ('git pull' in a git bash console in hbdet directory)

##############################################################################

import yaml

with open('simple_config.yml', 'r') as f:
    simple = yaml.safe_load(f)
    
with open('advanced_config.yml', 'r') as f:
    advanced = yaml.safe_load(f)
    
####################  AUDIO PROCESSING PARAMETERS  ###########################
## GLOBAL AUDIO PROCESSING PARAMETERS
SR = advanced['sample_rate']
## MEL-SPECTROGRAM PARAMETERS

# FFT window length
STFT_FRAME_LEN = advanced['stft_frame_len']

# number of fft bins for mel spectrogram
N_FREQ_BINS = advanced['number_of_frequency_bins']

## CALCULATION OF CONTEXT WINDOW LENGTH
# calculation of context window in seconds to fit stft frame length
# and number of freq bins. From the set time length, the stft frame length
# is subtracted, as it stays constant. The remainder gets used to calculate
# the module of that value and the freuqency bins - 1 -> this then gives the
# correct number of samples per context window excluding the stft length, 
# which is added subsequently.
set_length_samples = advanced['context_window_in_seconds']*SR
set_length_without_stft_frame = set_length_samples - STFT_FRAME_LEN
set_length_fixed = set_length_without_stft_frame % (N_FREQ_BINS-1)

CONTEXT_WIN = int(set_length_without_stft_frame - set_length_fixed 
                  + STFT_FRAME_LEN)

CONTEXT_WIN_S_CORRECTED = CONTEXT_WIN/SR

# downsample every audio file to this frame rate to ensure comparability
DOWNSAMPLE_SR = False

## Settings for Creation of Tfrecord Dataset
# limit of context windows in a tfrecords file
TFRECS_LIM = advanced['tfrecs_limit_per_file']
# train/test split
TRAIN_RATIO = advanced['train_ratio']
# test/val split
TEST_VAL_RATIO = advanced['test_val_ratio']

## Model Parameters
# threshold for predictions
THRESH = simple['thresh']
# prediction window limit
PRED_WIN_LIM = simple['prediction_window_limit']

# calculated global variables
FFT_HOP = (CONTEXT_WIN - STFT_FRAME_LEN) // (N_FREQ_BINS - 1)
PRED_BATCH_SIZE = PRED_WIN_LIM * CONTEXT_WIN   
           
## Paths
TFREC_DESTINATION = advanced['tfrecords_destination_folder']
ANNOT_DEST = simple['annotation_destination']
REV_ANNOT_SRC = simple['reviewed_annotation_source']
GEN_ANNOT_SRC = simple['generated_annotation_source']
SOUND_FILES_SOURCE = simple['sound_files_source']
GEN_ANNOTS_DIR = advanced['generated_annotations_folder']
# model directory
MODEL_DIR = 'hbdet/files/models'
# model name
MODEL_NAME = advanced['model_name']
TOP_DIR_NAME = advanced['top_dir_name']

#############  ANNOTATIONS  #####################################
DEFAULT_THRESH = advanced['default_threshold']
ANNOTATION_DF_FMIN = advanced['annotation_df_fmin']
ANNOTATION_DF_FMAX = advanced['annotation_df_fmax']
## Column Names
# column name for annotation prediction values
ANNOTATION_COLUMN = 'Prediction/Comments'


#################### RUN CONFIGURATION  ######################################
RUN_CONFIG = simple['run_config']
PRESET = simple['predefined_settings']

#################### TRAINING CONFIG  ########################################

MODELCLASSNAME = advanced['ModelClassName']
BATCH_SIZE = advanced['batch_size']
EPOCHS = advanced['epochs']
LOAD_CKPT_PATH = advanced['load_ckpt_path']
LOAD_G_CKPT = advanced['load_g_ckpt']
KERAS_MOD_NAME = advanced['keras_mod_name']
STEPS_PER_EPOCH = advanced['steps_per_epoch']
TIME_AUGS = advanced['time_augs']
MIXUP_AUGS = advanced['mixup_augs']
SPEC_AUG = advanced['spec_aug']
DATA_DESCRIPTION = advanced['data_description']
INIT_LR = float(advanced['init_lr'])
FINAL_LR = float(advanced['final_lr'])
PRE_BLOCKS = advanced['pre_blocks']
F_SCORE_BETA = advanced['f_score_beta']
F_SCORE_THRESH = advanced['f_score_thresh']
UNFREEZE = advanced['unfreeze']


##################### HOURLY PRESENCE DIR AND FILE NAMES #####################

HR_CNTS_SL = 'hourly_annotation_simple_limit'
HR_PRS_SL = 'hourly_presence_simple_limit'
HR_CNTS_SC = 'hourly_annotations_sequence_crit'
HR_PRS_SC = 'hourly_pres_sequ_crit'

# column name for daily annotations (cumulative counts)
HR_DA_COL = 'daily_annotations'
# column name for daily presence (binary)
HR_DP_COL = 'Daily_Presence'

