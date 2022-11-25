import pandas as pd
import numpy as np
from hbdet.funcs import get_files, get_datetime_from_filename
import hbdet.global_config as conf

def return_hourly_presence(df):
    if len(df) > 10:
        return 1
    else:
        return 0

df = pd.DataFrame(columns=['Date', 'Daily_Presence', 
                           *['%.2i' % i for i in np.arange(24)]])

files = get_files(location=conf.ANNOTATION_SOURCE, 
                  search_str='**/*txt')
files.sort()
dates = list(map(lambda x: get_datetime_from_filename(x.stem.split('_annot')[0]), files))
date_hour_tuple = list(map(lambda x: (str(x.date()), '%.2i'%int(x.hour)), dates))
tup, counts = np.unique(date_hour_tuple, axis=0, return_counts=True)
file_ind, row = 0, 0
for (date, hour), count in zip(tup, counts):
    annot = pd.DataFrame()
    for _ in range(count):
        annot = pd.concat([annot, pd.read_csv(files[file_ind], sep='\t')])
        file_ind += 1
    if not date in df['Date'].values:
        row += 1
        df.loc[row, 'Date'] = date
    df.loc[row, hour] = return_hourly_presence(annot)
    print(f'{file_ind}/{len(files)}')

df.to_csv('hourly_presence.csv')