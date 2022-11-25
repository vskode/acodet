import pandas as pd
import numpy as np
from hbdet.funcs import get_files, get_dt_filename
import hbdet.global_config as conf
from pathlib import Path

def return_hourly_presence(df, lim=10):
    if len(df) > lim:
        return 1
    else:
        return 0
    
def return_daily_presence(df):
    if 1 in df.loc[len(df), hours_of_day].values:
        return 1
    else:
        return 0
    
def get_val(path):
    df = pd.read_csv(path)
    return df
    
# thresh = 0.8
# lim = 6
for thresh in [0.7, 0.8, 0.9]:
    for lim in [3, 6, 9, 12, 15, 20]:


        incorrect = 0
        df_val = get_val('../Data/EL1.csv')
        pred_col = 'Prediction/Comments'
        hours_of_day = ['%.2i:00' % i for i in np.arange(24)]
        df = pd.DataFrame(columns=['Date', 'Daily_Presence', *hours_of_day])
        df_counts = pd.DataFrame(columns=['Date', 'daily_annotations', *hours_of_day])

        files = get_files(location=conf.ANNOTATION_DESTINATION, 
                        search_str='**/*txt')
        files.sort()
        dates = list(map(lambda x: get_dt_filename(x.stem.split('_annot')[0]), 
                        files))
        date_hour_tuple = list(map(lambda x: (str(x.date()), '%.2i:00'%int(x.hour)), dates))
        tup, counts = np.unique(date_hour_tuple, axis=0, return_counts=True)
        file_ind, row = 0, 0
        for (date, hour), count in zip(tup, counts):
            annot = pd.DataFrame()
            for _ in range(count):
                annot = pd.concat([annot, pd.read_csv(files[file_ind], sep='\t')])
                file_ind += 1
            annot = annot.loc[annot[pred_col] >= thresh]
            if not date in df['Date'].values:
                if not row == 0:
                    df.loc[row, 'Daily_Presence'] = return_daily_presence(df)
                    df_counts.loc[row, 'daily_annotations'] = sum(df_counts.loc[len(df_counts), hours_of_day].values)

                row += 1
                df.loc[row, 'Date'] = date
                df_counts.loc[row, 'Date'] = date
            df.loc[row, hour] = return_hourly_presence(annot, lim=lim)
            if df.loc[row, hour] != df_val.loc[row-1, hour]:
                df.loc[row, hour] = -return_hourly_presence(annot, lim=lim)
                df_counts.loc[row, hour] = -len(annot)
                incorrect += 1
            else:
                df_counts.loc[row, hour] = len(annot)

            # print(f'{file_ind}/{len(files)}')
        print('l:', lim, 'th:', thresh, 
            'incorrect:', incorrect, '%.2f' % (incorrect/(46*24)*100))
        # df.to_csv(Path(conf.ANNOTATION_DESTINATION).joinpath('hourly_presence.csv'))
        # df_counts.to_csv(Path(conf.ANNOTATION_DESTINATION).joinpath('hourly_annotation_counts.csv'))