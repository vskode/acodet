import pandas as pd
import numpy as np
from hbdet.funcs import get_files, get_dt_filename
import hbdet.global_config as conf
from pathlib import Path
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
sns.set_theme()
sns.set_style('white')
import time
time_start = time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())


def hourly_prs(df, lim=10):
    if len(df) > lim:
        return 1
    else:
        return 0
    
def daily_prs(df):
    if 1 in df.loc[len(df), h_of_day_str()].values:
        return 1
    else:
        return 0
    
def get_val(path):
    df = pd.read_csv(path)
    return df

def seq_crit(annot, n_prec_preds=conf.SC_CON_WIN, thresh_sc=0.9,
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
            
def h_of_day_str():
    return ['%.2i:00' % i for i in np.arange(24)]

def find_thresh05_path_in_dir(time_dir):
    """
    Get corrects paths leading to thresh_0.5 directory contatining annotations.
    Correct for incorrect paths, that already contain the thresh_0.5 path.

    Parameters
    ----------
    time_dir : str
        if run.py is run directly, a path to a specific timestamp can be passed

    Returns
    -------
    pathlib.Path
        correct path leading to thresh_0.5 directory
    """
    root = Path(conf.GEN_ANNOT_SRC)
    if root.parts[-1] == 'thresh_0.5':
        root = root.parent
    elif root.parts[-1] == 'thresh_0.9':
        root = root.parent
        
    if not time_dir:
        path = root.joinpath('thresh_0.5')
    else:
        path = root.joinpath(time_dir).joinpath('thresh_0.5')
    return path
    
# TODO fall einbauen wenn datei über mehrere stundengrenzen geht
    # datei nach zeitstunden splitten
def compute_hourly_pres(time_dir=None, 
                        thresh=conf.THRESH, 
                        lim=conf.SIMPLE_LIMIT, 
                        thresh_sc=conf.SC_THRESH, 
                        lim_sc=conf.SC_LIMIT, 
                        sc=False):
    path = find_thresh05_path_in_dir(time_dir)
    
    for dir in path.iterdir():
        if not dir.is_dir():
            continue
        df = pd.DataFrame(columns=['Date', conf.HR_DP_COL, *h_of_day_str()])
        df_sc = df.copy()
        df_counts = pd.DataFrame(columns=['Date', conf.HR_DA_COL, 
                                        *h_of_day_str()])
        df_sc_counts = df_counts.copy()

        files = get_files(location=path.joinpath(dir.stem), search_str='**/*txt')
        files.sort()
        
        dates = list(map(lambda x: get_dt_filename(x.stem.split('_annot')[0]), 
                        files))
        
        date_hour_tuple = list(map(lambda x: (str(x.date()), 
                                            '%.2i:00' % int(x.hour)), dates))
        
        tup, counts = np.unique(date_hour_tuple, axis=0, return_counts=True)
        
        file_ind, row = 0, 0
        for (date, hour), count in zip(tup, counts):
            annot_all = pd.DataFrame()
            
            for _ in range(count):
                annot_all = pd.concat([annot_all, pd.read_csv(files[file_ind], 
                                                            sep='\t')])
                file_ind += 1 
                
            if len(annot_all) == 0:
                end = False
            else:
                end = int(annot_all['End Time (s)'].iloc[-1])       
                
            for h in range(0, end or 1, 3600):
                annot = annot_all.loc[h < annot_all['Begin Time (s)']]
                annot = annot.loc[annot['Begin Time (s)'] < h+3600]
                if h > 0:
                    new_dt = (dt.datetime.strptime(date+hour, '%Y-%m-%d%H:00')
                              +dt.timedelta(hours=1))
                    date = str(new_dt.date())
                    hour = '%.2i:00' % new_dt.hour
                    
                        
            
                annot = annot.loc[annot[conf.ANNOTATION_COLUMN] >= thresh]
                # TODO hourly presence für dateien die über eine stunde lang sind
                if not date in df['Date'].values:
                    if not row == 0:
                        df.loc[row, conf.HR_DP_COL] = daily_prs(df)
                        
                        df_counts.loc[row, conf.HR_DA_COL] = sum(
                            df_counts.loc[len(df_counts), h_of_day_str()].values)
                        
                        if sc:
                            df_sc.loc[row, conf.HR_DP_COL] = daily_prs(df_sc)
                        
                            df_sc_counts.loc[row, conf.HR_DA_COL] = sum(
                                df_sc_counts.loc[len(df_sc_counts), 
                                                h_of_day_str()].values)
                            
                    row += 1
                    df.loc[row, 'Date'] = date
                    df_counts.loc[row, 'Date'] = date
                    if sc:
                        df_sc.loc[row, 'Date'] = date
                        df_sc_counts.loc[row, 'Date'] = date
                
                df.loc[row, hour] = hourly_prs(annot, lim=lim)
                df_counts.loc[row, hour] = len(annot)
                
                if file_ind == len(files):
                    df.loc[row, conf.HR_DP_COL] = daily_prs(df)
                    
                    df_counts.loc[row, conf.HR_DA_COL] = sum(
                        df_counts.loc[len(df_counts), h_of_day_str()].values)
                    
                    if sc:
                        df_sc.loc[row, conf.HR_DP_COL] = daily_prs(df_sc)
                        
                        df_sc_counts.loc[row, conf.HR_DA_COL] = sum(
                            df_sc_counts.loc[len(df_sc_counts), h_of_day_str()].values)
                    
                
                if sc:
                    df_sc_counts.loc[row, hour] = seq_crit(annot, thresh_sc=thresh_sc,
                                                        n_exceed_thresh=lim_sc)
                    df_sc.loc[row, hour] = int(bool(df_sc_counts.loc[row, hour]))

            print(f'Computing files in {dir.stem}: '
                  f'{file_ind}/{len(files)}', end='\r')
                    
        df.to_csv(get_path(path.joinpath(dir.stem), conf.HR_PRS_SL))
        df_counts.to_csv(get_path(path.joinpath(dir.stem), conf.HR_CNTS_SL))
        for metric in (conf.HR_CNTS_SL, conf.HR_PRS_SL):
            plot_hp(path.joinpath(dir.stem), lim, thresh, metric)
            
        if sc:
            df_sc.to_csv(get_path(path.joinpath(dir.stem), conf.HR_PRS_SC))
            df_sc_counts.to_csv(get_path(path.joinpath(dir.stem), conf.HR_CNTS_SC))
            for metric in (conf.HR_CNTS_SC, conf.HR_PRS_SC):
                plot_hp(path.joinpath(dir.stem), lim_sc, thresh_sc, metric)
        print('\n')

def get_path(path, metric): 
    save_path = Path(path).joinpath('analysis')
    save_path.mkdir(exist_ok=True, parents=True)
    return save_path.joinpath(f'{metric}.csv')

def get_title(metric):
    if 'annotation' in metric:
        return 'Annotation counts for each hour'
    elif 'presence' in metric:
        return 'Hourly presence'
        

def plot_hp(path, lim, thresh, metric):
    df = pd.read_csv(get_path(path, metric))
    h_pres = df.loc[:, h_of_day_str()]
    h_pres.index = df['Date']
    plt.figure(figsize=[8, 6])
    plt.title(f'{get_title(metric)}, limit={lim:.0f}, '
              f'threshold={thresh:.2f}')
    if 'presence' in metric:
        d = {'vmin': 0, 'vmax': 1}
    else:
        d = {'vmax': conf.HR_CNTS_VMAX}
    sns.heatmap(h_pres.T, cmap='crest', **d)
    plt.ylabel('hour of day')
    plt.tight_layout()
    path = Path(path).joinpath('analysis')
    plt.savefig(path.joinpath(f'{metric}_{thresh:.2f}_{lim:.0f}.png'),
                dpi = 150)
    plt.close()

def plot_varying_limits(annotations_path=conf.ANNOT_DEST):
    thresh_sl, thresh_sc = 0.9, 0.9
    for lim_sl, lim_sc in zip(np.linspace(10, 48, 20), np.linspace(1, 20, 20)):
        for lim, thresh in zip((lim_sl, thresh_sl), (lim_sc, thresh_sc)):
            compute_hourly_pres(annotations_path, thresh_sc=thresh_sc, 
                                lim_sc=lim_sc, thresh=thresh, lim=lim)
            for metric in (conf.HR_CNTS_SC, conf.HR_CNTS_SL,
                           conf.HR_PRS_SC, conf.HR_PRS_SL):
                plot_hp(annotations_path, lim, thresh, metric)