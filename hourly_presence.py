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
time_start = time.strftime('%Y-%m-%d_%H_%M', time.gmtime())


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

def sequence_criterion(annot, n_prec_preds=20, thresh_sc=0.9,
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
def main(path, thresh=0.94, lim=10, thresh_sc=0.85, lim_sc=4, sc=True):
    df_val = get_val(f"../Data/SAMOSAS_val/{Path(path).stem.split('_')[-1]}.csv")
    pred_col = 'Prediction/Comments'
    hours_of_day = ['%.2i:00' % i for i in np.arange(24)]
    df = pd.DataFrame(columns=['Date', 'Daily_Presence', *h_of_day_str()])
    df_counts = pd.DataFrame(columns=['Date', 'daily_annotations', 
                                      *h_of_day_str()])
    df_sc = pd.DataFrame(columns=['Date', 'daily_annotations', *h_of_day_str()])
    df_sc_counts = pd.DataFrame(columns=['Date', 'daily_annotations', 
                                         *h_of_day_str()])

    files = get_files(location=path, 
                    search_str='**/*txt')
    files.sort()
    dates = list(map(lambda x: get_dt_filename(x.stem.split('_annot')[0]), 
                    files))
    date_hour_tuple = list(map(lambda x: (str(x.date()), '%.2i:00'%int(x.hour)), dates))
    tup, counts = np.unique(date_hour_tuple, axis=0, return_counts=True)
    file_ind, row = 0, 0
    incorrect, sc_incorrect = 0, 0
    for (date, hour), count in zip(tup, counts):
        annot_all = pd.DataFrame()
        
        for _ in range(count):
            annot_all = pd.concat([annot_all, pd.read_csv(files[file_ind], sep='\t')])
            file_ind += 1
            
        annot = annot_all
        
            
        annot = annot.loc[annot[pred_col] >= thresh]
        if not date in df['Date'].values:
            if not row == 0:
                df.loc[row, 'Daily_Presence'] = return_daily_presence(df, hours_of_day)
                df_counts.loc[row, 'daily_annotations'] = sum(df_counts.loc[len(df_counts), 
                                                                            hours_of_day].values)
                if sc:
                    df_sc.loc[row, 'Daily_Presence'] = return_daily_presence(df_sc, hours_of_day)
                    df_sc_counts.loc[row, 'daily_annotations'] = sum(df_sc_counts.
                                                                     loc[len(df_sc_counts), 
                                                                         hours_of_day].values)
                    
            
            row += 1
            df.loc[row, 'Date'] = date
            df_counts.loc[row, 'Date'] = date
            if sc:
                df_sc.loc[row, 'Date'] = date
                df_sc_counts.loc[row, 'Date'] = date
        
        df.loc[row, hour] = return_hourly_presence(annot, lim=lim)
        df_counts.loc[row, hour] = len(annot)
        if file_ind == len(files):
            df.loc[row, 'Daily_Presence'] = return_daily_presence(df, hours_of_day)
            df_counts.loc[row, 'daily_annotations'] = sum(df_counts.loc[len(df_counts), 
                                                                        hours_of_day].values)
            if sc:
                df_sc.loc[row, 'Daily_Presence'] = return_daily_presence(df_sc, hours_of_day)
                df_sc_counts.loc[row, 'daily_annotations'] = sum(df_sc_counts.
                                                                    loc[len(df_sc_counts), 
                                                                        hours_of_day].values)
            
        
        if sc:
            df_sc_counts.loc[row, hour] = sequence_criterion(annot, thresh_sc=thresh_sc,
                                                                n_exceed_thresh=lim_sc)
            df_sc.loc[row, hour] = int(bool(df_sc_counts.loc[row, hour]))
        
        
        if df.loc[row, hour] != df_val.loc[row-1, hour]:
            incorrect += 1
        if sc and df_sc.loc[row, hour] != df_val.loc[row-1, hour]:
            sc_incorrect += 1


        # print(f'{file_ind}/{len(files)}')
            
    print('l:', lim, 'th:', thresh, 
        'incorrect:', incorrect, '%.2f' % (incorrect/(46*24)*100))
    print('l:', lim_sc, 'th:', thresh_sc, 
            'sc_incorrect:', sc_incorrect, '%.2f' % (sc_incorrect/(46*24)*100))
    df.to_csv(Path(path).joinpath('analysis/hourly_presence.csv'))
    df_counts.to_csv(Path(path).joinpath('analysis/hourly_annotation_counts.csv'))
    if sc:
        df_sc.to_csv(Path(path).joinpath('analysis/hourly_pres_sequ_crit.csv'))
        df_sc_counts.to_csv(Path(path).joinpath('analysis/hourly_pres_sequ_crit_counts.csv'))

# main()
def plot_counts_per_hour(path, lim, thresh):
    df = pd.read_csv(Path(path).joinpath('analysis/hourly_annotation_counts.csv'))
    h_pres = df.loc[:, h_of_day_str()]
    h_pres.index = df['Date']
    plt.figure()
    plt.title(f'Annotation counts for each hour, limit={lim:.0f}, threshold={thresh:.2f}')
    sns.heatmap(h_pres.T, cmap='crest')
    plt.ylabel('hour of day')
    plt.tight_layout()
    path = Path(path).joinpath(f'analysis/{time_start}_hourly_counts_plots')
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(path).joinpath('hourly_counts_plot.png'))
    plt.close()

def plot_hourly_presence(path, lim, thresh):
    df = pd.read_csv(Path(path).joinpath('analysis/hourly_presence.csv'))
    h_pres = df.loc[:, h_of_day_str()]
    h_pres.index = df['Date']
    plt.figure()
    plt.title(f'Annotation counts for each hour, limit={lim:.0f}, threshold={thresh:.2f}')
    sns.heatmap(h_pres.T, cmap='crest')
    plt.ylabel('hour of day')
    plt.tight_layout()
    path = Path(path).joinpath(f'analysis/{time_start}_hourly_pres_plots')
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(path).joinpath('hourly_presence_plot.png'))
    plt.close()

def plot_counts_per_hour_sc(path, lim, thresh):
    df = pd.read_csv(Path(path).joinpath('analysis/hourly_pres_sequ_crit_counts.csv'))
    h_pres = df.loc[:, h_of_day_str()]
    h_pres.index = df['Date']
    plt.figure()
    plt.title(f'Annotation counts for each hour, limit={lim:.0f}, threshold={thresh:.2f}')
    sns.heatmap(h_pres.T, cmap='crest')
    plt.ylabel('hour of day')
    plt.tight_layout()
    path = Path(path).joinpath(f'analysis/{time_start}_hourly_counts_plots_sc')
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(path).joinpath('hourly_counts_sc_plot.png'))
    plt.close()

def plot_hourly_presence_sc(path, lim, thresh):
    df = pd.read_csv(Path(path).joinpath('analysis/hourly_pres_sequ_crit.csv'))
    h_pres = df.loc[:, h_of_day_str()]
    h_pres.index = df['Date']
    plt.figure()
    plt.title(f'Annotation counts for each hour, limit={lim:.0f}, threshold={thresh:.2f}')
    sns.heatmap(h_pres.T, cmap='crest')
    plt.ylabel('hour of day')
    plt.tight_layout()
    path = Path(path).joinpath(f'analysis/{time_start}_hourly_pres_plots_sc')
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(path).joinpath('hourly_presence_sc_plot.png'))
    plt.close()
    
def plot_validation(path):
    df = pd.read_csv(path)
    h_pres = df.loc[:, h_of_day_str()]
    h_pres.index = df['Date']
    plt.figure()
    plt.title('Reviewed Hourly Presence')
    sns.heatmap(h_pres.T, cmap='crest')
    plt.ylabel('hour of day')
    plt.tight_layout()
    plt.savefig(Path(path).parent.joinpath(f'validation_hourly_presence_{Path(path).stem}.png'))
    plt.close()
    
def plot_comparison(pred, val, thresh, lim):
    df_p = pd.read_csv(Path(pred).joinpath('analysis/hourly_presence.csv'))
    df_v = pd.read_csv(val)
    df_c = (df_p.loc[:, [*h_of_day_str()]] 
            - df_v.loc[:, [*h_of_day_str()]])
    df_c.index = df_p['Date']    
    prec = np.sum(np.sum(abs(df_c[df_c != 0])))/(df_c.shape[0]*df_c.shape[1])
    FNr = np.sum(np.sum(abs(df_c[df_c == -1])))/(df_c.shape[0]*df_c.shape[1])
    FPr = np.sum(np.sum(abs(df_c[df_c == 1])))/(df_c.shape[0]*df_c.shape[1])
    print(prec)
    plt.figure()
    plt.title(f'Difference in Hourly Presence, FNr={abs(FNr):.2f}, FPr={FPr:.2f}')
    sns.heatmap(df_c.T, vmin=-1, vmax=1, cmap='crest')
    plt.ylabel('hour of day')
    plt.tight_layout()
    path = Path(pred).joinpath(f'analysis/{time_start}')
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.joinpath(f'{prec:.2f}_diff_hourly_presence_{thresh:.2f}_{lim:.0f}.png'))
    plt.close()

def plot_comparison_sc(pred, val, thresh, lim):
    df_p = pd.read_csv(Path(pred).joinpath('analysis/hourly_pres_sequ_crit.csv'))
    df_v = pd.read_csv(val)
    df_c = (df_p.loc[:, [*h_of_day_str()]] 
            - df_v.loc[:, [*h_of_day_str()]])
    df_c.index = df_p['Date']    
    prec = np.sum(np.sum(df_c[df_c != 0]))/(df_c.shape[0]*df_c.shape[1])
    FNr = np.sum(np.sum(df_c[df_c == -1]))/(df_c.shape[0]*df_c.shape[1])
    FPr = np.sum(np.sum(df_c[df_c == 1]))/(df_c.shape[0]*df_c.shape[1])
    print(prec)
    plt.figure()
    plt.title(f'Difference in Hourly Presence, FNr={abs(FNr):.2f}, FPr={FPr:.2f}')
    sns.heatmap(df_c.T, vmin=-1, vmax=1, cmap='crest')
    plt.ylabel('hour of day')
    plt.tight_layout()
    path = Path(pred).joinpath(f'analysis/{time_start}_sc')
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path.joinpath(f'{prec:.2f}_diff_hourly_presence_sc_{thresh:.2f}_{lim:.0f}.png'))
    plt.close()
    
# TODO csv mit negativen und ohne machen und dann ein plot wo di differenz 
# zwischen val und preds gezeigt wird - auch noch für N1 und S1
# plot_counts_per_hour('../generated_annotations/2022-12-01_13_EL1')
# plot_hourly_presence('../generated_annotations/2022-12-01_13_EL1')
# plot_validation('../Data/SAMOSAS_val/S1.csv')
# for thresh in np.linspace(0.85, 0.95, 11):
#     for lim in np.linspace(12, 33, 8):

# for thresh_sc in np.linspace(0.89, 0.95, 7):
#     for lim_sc in np.linspace(4, 8, 5):
for thresh, thresh_sc in zip(np.linspace(0.83, 0.95, 7), np.linspace(0.83, 0.95, 7)):
    for lim, lim_sc in zip(np.linspace(6, 36, 5), np.linspace(3, 7, 5)):
        for s in ['S1', 'EL1', 'N1']:
            main(f'../generated_annotations/2022-11-30_11_{s}', thresh_sc=thresh_sc, lim_sc=lim_sc,
                                                                thresh=thresh, lim=lim)
            plot_comparison(f'../generated_annotations/2022-11-30_11_{s}', 
                               f'../Data/SAMOSAS_val/{s}.csv', thresh, lim)
            plot_comparison_sc(f'../generated_annotations/2022-11-30_11_{s}', 
                            f'../Data/SAMOSAS_val/{s}.csv', thresh_sc, lim_sc)
            plot_counts_per_hour(f'../generated_annotations/2022-11-30_11_{s}', lim, thresh)
            plot_hourly_presence(f'../generated_annotations/2022-11-30_11_{s}', lim, thresh)
            plot_counts_per_hour_sc(f'../generated_annotations/2022-11-30_11_{s}', lim_sc, thresh_sc)
            plot_hourly_presence_sc(f'../generated_annotations/2022-11-30_11_{s}', lim_sc, thresh_sc)