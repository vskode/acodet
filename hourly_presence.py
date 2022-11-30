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
    
def return_daily_presence(df, hours_of_day):
    if 1 in df.loc[len(df), hours_of_day].values:
        return 1
    else:
        return 0
    
def get_val(path):
    df = pd.read_csv(path)
    return df

def sequence_criterion(annot, n_prec_preds=15, thresh_sc=0.9,
                       n_exceed_thresh=4):
    sequ_crit = 0
    annot = annot.loc[annot[conf.ANNOTATION_COLUMN] >= thresh_sc]
    for i, row in annot.iterrows():
        bool1 = 0 < (row['Begin Time (s)'] - annot['Begin Time (s)'])
        bool2 = ((row['Begin Time (s)'] - annot['Begin Time (s)']) 
                 < n_prec_preds * conf.CONTEXT_WIN/conf.SR)
        prec_anns = annot.loc[bool1 * bool2]
        if len(prec_anns) > n_exceed_thresh:
            sequ_crit += 1
    return sequ_crit
            
        
        # TODO vollenden
    
# TODO kriterium einbauen mit aufeinanderfolgneden 0.8 preds
# TODO fall einbauen wenn datei über mehrere stundengrenzen geht
    # datei nach zeitstunden splitten
# TODO leere zellen füllen
thresh = 0.94
lim = 10
# for thresh in np.linspace(0.85, 0.95, 11):
#     for lim in np.linspace(4, 44, 21):
# thresh_sc = 0.85
# lim_sc = 4
# for thresh_sc in [0.85, 0.86, 0.87, 0.88, 0.89]:
#     for lim_sc in [1, 2, 3, 4, 5]:

sc_incorrect, incorrect = 0, 0
df_val = get_val('../Data/EL1.csv')
pred_col = 'Prediction/Comments'
hours_of_day = ['%.2i:00' % i for i in np.arange(24)]
df = pd.DataFrame(columns=['Date', 'Daily_Presence', *hours_of_day])
df_counts = pd.DataFrame(columns=['Date', 'daily_annotations', *hours_of_day])
df_sc = pd.DataFrame(columns=['Date', 'daily_annotations', *hours_of_day])
df_sc_counts = pd.DataFrame(columns=['Date', 'daily_annotations', *hours_of_day])

files = get_files(location='../generated_annotations/2022-11-30_11', 
                search_str='**/*txt')
files.sort()
dates = list(map(lambda x: get_dt_filename(x.stem.split('_annot')[0]), 
                files))
date_hour_tuple = list(map(lambda x: (str(x.date()), '%.2i:00'%int(x.hour)), dates))
tup, counts = np.unique(date_hour_tuple, axis=0, return_counts=True)
file_ind, row = 0, 0
for (date, hour), count in zip(tup, counts):
    annot_all = pd.DataFrame()
    
    for _ in range(count):
        annot_all = pd.concat([annot_all, pd.read_csv(files[file_ind], sep='\t')])
        file_ind += 1
    # for hrs in range(int(annot_all['End Time (s)'].iloc[-1]/3600) + 1):
    annot = annot_all
    
        
    annot = annot.loc[annot[pred_col] >= thresh]
    if not date in df['Date'].values:
        if not row == 0:
            df.loc[row, 'Daily_Presence'] = return_daily_presence(df, hours_of_day)
            df_counts.loc[row, 'daily_annotations'] = sum(df_counts.loc[len(df_counts), hours_of_day].values)
            # df_sc.loc[row, 'Daily_Presence'] = return_daily_presence(df_sc, hours_of_day)
            # df_sc_counts.loc[row, 'daily_annotations'] = sum(df_sc_counts.loc[len(df_sc_counts), hours_of_day].values)

        row += 1
        df.loc[row, 'Date'] = date
        df_counts.loc[row, 'Date'] = date
        df_sc.loc[row, 'Date'] = date
        df_sc_counts.loc[row, 'Date'] = date
    
    df.loc[row, hour] = return_hourly_presence(annot, lim=lim)
    df_counts.loc[row, hour] = len(annot)
    if df.loc[row, hour] != df_val.loc[row-1, hour]:
        df.loc[row, hour] = -df.loc[row, hour]
        df_counts.loc[row, hour] = -df_counts.loc[row, hour]
        incorrect += 1
        
    # df_sc_counts.loc[row, hour] = sequence_criterion(annot, thresh_sc=thresh_sc,
    #                                                     n_exceed_thresh=lim_sc)
    # df_sc.loc[row, hour] = int(bool(df_sc_counts.loc[row, hour]))
    # if df_sc.loc[row, hour] != df_val.loc[row-1, hour]:
    #     df_sc_counts.loc[row, hour] = -df_sc_counts.loc[row, hour]
    #     df_sc.loc[row, hour] = -df_sc.loc[row, hour]-10
        # sc_incorrect += 1

    print(f'{file_ind}/{len(files)}')
        
print('l:', lim, 'th:', thresh, 
    'incorrect:', incorrect, '%.2f' % (incorrect/(46*24)*100))
# print('l:', lim_sc, 'th:', thresh_sc, 
#         'sc_incorrect:', sc_incorrect, '%.2f' % (sc_incorrect/(46*24)*100))
df.to_csv(Path(conf.ANNOTATION_DESTINATION).joinpath('hourly_presence.csv'))
df_counts.to_csv(Path(conf.ANNOTATION_DESTINATION).joinpath('hourly_annotation_counts.csv'))
df_sc.to_csv(Path(conf.ANNOTATION_DESTINATION).joinpath('hourly_pres_sequ_crit.csv'))
df_sc_counts.to_csv(Path(conf.ANNOTATION_DESTINATION).joinpath('hourly_pres_sequ_crit_counts.csv'))
