import numpy as np
import pandas as pd
from hbdet.tfrec import write_tfrecords, exclude_files_from_dataset

annots = pd.read_csv('Daten/ket_annot.csv')
annots, annots_poor = exclude_files_from_dataset(annots)

for shift in [0]:
    write_tfrecords(annots, shift = shift)
    # write_tfrecords(annots_poor, shift = shift, alt_subdir = 'noisy')