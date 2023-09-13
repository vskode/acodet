##############################################################################

# THIS FILE IS ONLY MEANT TO BE EDITED IF YOU ARE SURE!

# PROGRAM FAILURE IS LIKELY TO OCCUR IF YOU ARE UNSURE OF THE
# CONSEQUENCES OF YOUR CHANGES.

# IF YOU HAVE CHANGED VALUES AND ARE ENCOUNTERING ERRORS,
# STASH YOUR CHANGES ('git stash' in a git bash console in acodet directory)
# AND THEN PULL AGAIN ('git pull' in a git bash console in acodet directory)

##############################################################################

import json
import streamlit as st

if "session_started" in st.session_state:
    session = {**st.session_state}
else:
    with open("acodet/src/tmp_session.json", "r") as f:
        session = json.load(f)

####################  AUDIO PROCESSING PARAMETERS  ###########################
## GLOBAL AUDIO PROCESSING PARAMETERS
SR = session["sample_rate"]
## MEL-SPECTROGRAM PARAMETERS

# FFT window length
STFT_FRAME_LEN = session["stft_frame_len"]

# number of fft bins for mel spectrogram
N_FREQ_BINS = session["number_of_frequency_bins"]

## CALCULATION OF CONTEXT WINDOW LENGTH
# calculation of context window in seconds to fit stft frame length
# and number of freq bins. From the set time length, the stft frame length
# is subtracted, as it stays constant. The remainder gets used to calculate
# the module of that value and the freuqency bins - 1 -> this then gives the
# correct number of samples per context window excluding the stft length,
# which is added subsequently.
set_length_samples = session["context_window_in_seconds"] * SR
set_length_without_stft_frame = set_length_samples - STFT_FRAME_LEN
set_length_fixed = set_length_without_stft_frame % (N_FREQ_BINS - 1)

CONTEXT_WIN = int(
    set_length_without_stft_frame - set_length_fixed + STFT_FRAME_LEN
)

CONTEXT_WIN_S_CORRECTED = CONTEXT_WIN / SR

# downsample every audio file to this frame rate to ensure comparability
DOWNSAMPLE_SR = False

## Settings for Creation of Tfrecord Dataset
# limit of context windows in a tfrecords file
TFRECS_LIM = session["tfrecs_limit_per_file"]
# train/test split
TRAIN_RATIO = session["train_ratio"]
# test/val split
TEST_VAL_RATIO = session["test_val_ratio"]

## Model Parameters
# threshold for predictions
THRESH = session["thresh"]
# simple limit for hourly presence
SIMPLE_LIMIT = session["simple_limit"]
# sequence criterion threshold
SEQUENCE_THRESH = session["sequence_thresh"]
# sequence criterion limit
SEQUENCE_LIMIT = session["sequence_limit"]
# number of consecutive winodws for sequence criterion
SEQUENCE_CON_WIN = session["sequence_con_win"]
# limit for colorbar for hourly annotations
HR_CNTS_VMAX = session["max_annots_per_hour"]
# prediction window limit
PRED_WIN_LIM = session["prediction_window_limit"]

# calculated global variables
FFT_HOP = (CONTEXT_WIN - STFT_FRAME_LEN) // (N_FREQ_BINS - 1)
PRED_BATCH_SIZE = PRED_WIN_LIM * CONTEXT_WIN

## Paths
TFREC_DESTINATION = session["tfrecords_destination_folder"]
ANNOT_DEST = session["annotation_destination"]
REV_ANNOT_SRC = session["reviewed_annotation_source"]
GEN_ANNOT_SRC = session["generated_annotation_source"]
SOUND_FILES_SOURCE = session["sound_files_source"]
GEN_ANNOTS_DIR = session["generated_annotations_folder"]
# model directory
MODEL_DIR = "acodet/src/models"
# model name
MODEL_NAME = session["model_name"]
TOP_DIR_NAME = session["top_dir_name"]

#############  ANNOTATIONS  #####################################
DEFAULT_THRESH = session["default_threshold"]
ANNOTATION_DF_FMIN = session["annotation_df_fmin"]
ANNOTATION_DF_FMAX = session["annotation_df_fmax"]
## Column Names
# column name for annotation prediction values
ANNOTATION_COLUMN = "Prediction/Comments"


#################### RUN CONFIGURATION  ######################################
RUN_CONFIG = session["run_config"]
PRESET = session["predefined_settings"]

#################### TRAINING CONFIG  ########################################

MODELCLASSNAME = session["ModelClassName"]
BATCH_SIZE = session["batch_size"]
EPOCHS = session["epochs"]
LOAD_CKPT_PATH = session["load_ckpt_path"]
LOAD_G_CKPT = session["load_g_ckpt"]
KERAS_MOD_NAME = session["keras_mod_name"]
STEPS_PER_EPOCH = session["steps_per_epoch"]
TIME_AUGS = session["time_augs"]
MIXUP_AUGS = session["mixup_augs"]
SPEC_AUG = session["spec_aug"]
DATA_DESCRIPTION = session["data_description"]
INIT_LR = float(session["init_lr"])
FINAL_LR = float(session["final_lr"])
PRE_BLOCKS = session["pre_blocks"]
F_SCORE_BETA = session["f_score_beta"]
F_SCORE_THRESH = session["f_score_thresh"]
UNFREEZE = session["unfreeze"]


##################### HOURLY PRESENCE DIR AND FILE NAMES #####################

HR_CNTS_SL = "hourly_annotation_simple_limit"
HR_PRS_SL = "hourly_presence_simple_limit"
HR_CNTS_SC = "hourly_annotation_sequence_limit"
HR_PRS_SC = "hourly_presence_sequence_limit"
HR_VAL_PATH = session["hourly_presence_validation_path"]

# column name for daily annotations (cumulative counts)
HR_DA_COL = "daily_annotations"
# column name for daily presence (binary)
HR_DP_COL = "Daily_Presence"

STREAMLIT = session["streamlit"]
