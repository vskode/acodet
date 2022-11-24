import time
from hbdet.models import GoogleMod
from hbdet.funcs import get_files, gen_annotations
from hbdet import global_config as conf
    
if __name__ == '__main__':
    time_start = time.strftime('%Y-%m-%d_%H', time.gmtime())
    train_date = '2022-11-22_17'
    files = get_files(location=conf.SOUND_FILES_SOURCE,
                      search_str='**/*wav')
    # files = get_files(location='/media/vincent/Extreme SSD/MA/for_manual_annotation/src_to_be_annotated/resampled_2kHz',
    #                   search_str='**/*wav')
    for file in files:
        try:
            gen_annotations(file, GoogleMod, training_path='../trainings', 
                            mod_label=train_date, time_start=time_start)
        except Exception as e:
            print(f"{file} couldn't be loaded, continuing with next file.\n", e)
            continue