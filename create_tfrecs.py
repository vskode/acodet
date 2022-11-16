import numpy as np
import pandas as pd
import hbdet.tfrec as tfrecord

# file = 'Daten/Tim/2020-11-17/channelA_2020-11-17_00-00-04.wav'
# tfrecord.write_tfrecs_for_mixup(file)

annots = pd.read_csv('Daten/2022-11-11_Benoit_annots.csv')
# annots, annots_poor = tfrecord.exclude_files_from_dataset(annots)

for shift in [0]:
    tfrecord.write_tfrecords(annots, all_noise=False, inbetween_noise=True)