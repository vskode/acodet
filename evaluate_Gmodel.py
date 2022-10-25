from hbdet.plot_utils import create_and_save_figure
from hbdet.google_funcs import GoogleMod

tfrec_path = 'Daten/Datasets/ScotWest_v1_2khz/tfrecords_0s_shift'
train_date = '2022-10-21_15'
batch_size = 32
create_and_save_figure(GoogleMod, tfrec_path, batch_size, train_date, 
                        init_lr='1e-3', end_lr='1e-6', 
                        plot_cm=True, plot_pr=True)
