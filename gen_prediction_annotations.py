import time
from hbdet.google_funcs import GoogleMod
from hbdet.funcs import get_files, gen_annotations
    
if __name__ == '__main__':
    time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
    train_date = '2022-11-17_17'
    files = get_files(location='Daten/for_manual_annotation/src_to_be_annotated',
                      search_str='**/*wav')
    # files = get_files(location='Daten/for_manual_annotation/src/resampled_2kHz/Bundle3',
    #                   search_str='*')
    for file in files:
        try:
            gen_annotations(file, GoogleMod, training_path='trainings', 
                            mod_label=train_date, time_start=time_start)
        except:
            print(f"{file} couldn't be loaded, continuing with next file")
            continue