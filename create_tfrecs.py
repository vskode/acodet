import numpy as np
import pandas as pd
import hbdet.tfrec as tfrecord
from pathlib import Path
import hbdet.global_config as conf

# file = 'Daten/Tim/2020-11-17/channelA_2020-11-17_00-00-04.wav'
# tfrecord.write_tfrecs_for_mixup(file)

SOURCE_DIR = '../combined_annotations/2022-11-27_02'

annotation_files = Path(SOURCE_DIR).glob('**/*.csv')
files = list(annotation_files)
for file in files:
    annots = pd.read_csv(file)
    if 'explicit_noise' in file.stem:
        all_noise = True
    else:
        all_noise = False

    save_dir = (Path(conf.TFREC_DESTINATION)
                .joinpath(list(file.relative_to(SOURCE_DIR).parents)[-2]))
    
    tfrecord.write_tfrecords(annots, save_dir, 
                            all_noise=all_noise, inbetween_noise=False)