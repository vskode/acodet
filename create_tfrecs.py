from utils.tfrec import write_tfrecords, exclude_files_from_dataset
import pandas as pd
import numpy as np

annots = pd.read_csv('Daten/ket_annot.csv')
annots, annots_poor = exclude_files_from_dataset(annots)

for shift in [0, 0.5, 1, 1.5, 2]:
    write_tfrecords(annots, shift = shift)
    write_tfrecords(annots_poor, shift = shift, alt_subdir = 'noisy')