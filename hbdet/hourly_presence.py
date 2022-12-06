import pandas as pd
import numpy as np
from hbdet.funcs import get_files, get_dt_filename
import hbdet.global_config as conf
from pathlib import Path
import matplotlib.pyplot as plt
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

def seq_crit(annot, n_prec_preds=20, thresh_sc=0.9,
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
    
# TODO fall einbauen wenn datei über mehrere stundengrenzen geht
    # datei nach zeitstunden splitten
def compute_hourly_pres(time_dir=None, thresh=0.9, lim=7, thresh_sc=0.85, 
                        lim_sc=4, sc=False):
    if not time_dir:
        path = Path(conf.GEN_ANNOT_SRC).joinpath('thresh_0.5')
    else:
        path = (Path(conf.GEN_ANNOTS_DIR).joinpath(time_dir)
                .joinpath('thresh_0.5'))
    
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
            annot = annot_all
            
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

            print(f'{file_ind}/{len(files)}')
                    
        df.to_csv(get_path(path.joinpath(dir.stem), conf.HR_PRS_SL))
        df_counts.to_csv(get_path(path.joinpath(dir.stem), conf.HR_CNTS_SL))
        if sc:
            df_sc.to_csv(get_path(path.joinpath(dir.stem), conf.HR_PRS_SC))
            df_sc_counts.to_csv(get_path(path.joinpath(dir.stem), conf.HR_CNTS_SC))
        for metric in (conf.HR_CNTS_SL, conf.HR_PRS_SL):
            plot_hp(path.joinpath(dir.stem), lim, thresh, metric)

def get_path(path, metric): 
    save_path = Path(path).joinpath('analysis')
    save_path.mkdir(exist_ok=True, parents=True)
    return save_path.joinpath(f'{metric}.csv')

def plot_hp(path, lim, thresh, metric):
    df = pd.read_csv(get_path(path, metric))
    h_pres = df.loc[:, h_of_day_str()]
    h_pres.index = df['Date']
    plt.figure()
    plt.title(f'Annotation counts for each hour, limit={lim:.0f}, '
              f'threshold={thresh:.2f}')
    sns.heatmap(h_pres.T, cmap='crest')
    plt.ylabel('hour of day')
    plt.tight_layout()
    path = Path(path).joinpath(f'analysis/{time_start}_{metric}_dir')
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(path).joinpath(f'{metric}_{thresh:.2f}_{lim:.0f}.png'))
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