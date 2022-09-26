import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
import time
from pathlib import Path
import json

if not 'Daten' in os.listdir():
    os.chdir('../..')
    
def plot_performance_per_file():
    file_path = 'GoogleMod_model_evaluation.csv'
    df = pd.read_csv(file_path)


    quality_colors_dict = {'awful':'darkred', 
                        'v poor': 'indianred', 
                        'poor': 'red',
                        'fair':'orange', 
                        'good': 'gold', 
                        'v good': 'yellowgreen',
                        'unknown':'gray' }
    c = lambda x: quality_colors_dict[x]
    colors = list()
    dates = list()
    for index, row in df.iterrows():
        if Path(row.file).stem[0] == 'P':
            file_date = pd.to_datetime(Path(row.file).stem, 
                                format='PAM_%Y%m%d_%H%M%S_000')
        elif Path(row.file).stem[0] == 'c':
            file_date = pd.to_datetime(Path(row.file).stem.split('A_')[1],
                                        format='%Y-%m-%d_%H-%M-%S')
        else:
            file_date = pd.to_datetime(Path(row.file).stem.split('.')[1], 
                                        format='%y%m%d%H%M%S')
        dates.append(file_date)
        # if index < 3:
        #     dates.append( pd.to_datetime(Path(row.file).stem.split('.')[1], 
        #                                  format='%y%m%d%H%M%S') )
        # else:
        #     dates.append( pd.to_datetime(Path(row.file).stem, 
        #                                  format='PAM_%Y%m%d_%H%M%S_000') )
        colors.append(c(row.quality_of_recording))

    df['dates'] = dates
    df['dates'] = pd.to_datetime(df['dates']).dt.date


    fig, ax = plt.subplots(figsize = [14, 9])

    # ax.set_ylabel('MSE of predictions [%]')
    ax.set_ylabel('Binary cross-entropy')
    ax.set_title('Stanton Bank prediction accuracy | model = google'\
                    '\nnumber of annotations on bars')  
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels=df['dates'], rotation = 'vertical')

    pps = ax.bar(np.arange(len(df)), df['bin_cross_entr(GoogleMod)'], 
                0.5, color = colors)
    for i, p in enumerate(pps):
        height = p.get_height()
        ax.annotate(df.number_of_annots[i],
            xy=(p.get_x() + p.get_width() / 2, height),
            xytext=(0, 3), # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')
    patches = list()
    for key, value in quality_colors_dict.items():
        patches.append(mpatches.Patch(color = value, label=key))
    ax.legend(handles = patches, bbox_to_anchor=[1, 1])
    # plt.show()

    fig.savefig('google_pred_acc_bin_crossentrpy.png', dpi = 300, 
                facecolor = 'white')


def plot_model_results(datetime, **kwargs):

    fig, ax = plt.subplots(ncols = 4, nrows = 2, figsize = [15, 8])

    checkpoint_paths = Path(f"trainings/{datetime}").glob('unfreeze_*')
    for checkpoint_path in checkpoint_paths:
        unfreeze = int(checkpoint_path.stem.split('_')[-1])

        if not Path(f"{checkpoint_path}/results.json").exists():
            continue
        with open(f"{checkpoint_path}/results.json", 'r') as f:
            results = json.load(f)

        for i, m in enumerate(results.keys()):
            row = i // 4
            col = np.mod(i, 4)
            if row == 1 and col == 0:
                ax[row, col].set_ylim([0, 2])
            ax[row, col].plot(results[m], 
                            label = f'{unfreeze}')
            if row == 0:
                ax[row, col].set_title(f'{m}')
            if row == col == 0:
                ax[row, col].set_ylabel('training')
            elif row == 1 and col == 0:
                ax[row, col].set_ylabel('val')
    ax[0, 0].legend()

    info_string = ''
    for key, val in kwargs.items():
        info_string += f' | {key}: {val}'
    
    today = time.ctime()
    fig.suptitle(f'Model Results{info_string}'
                '\n'
                f'{today}')
    ref_time = time.strftime('%Y%m%d', time.gmtime())
    fig.savefig(f'trainings/{datetime}/model_results_{ref_time}.png')

if __name__ == '__main__':
    plot_model_results('2022-09-21_08', dataset = 'good and poor data, 5 shifts from 0s - 2s',
                                        begin_lr = '0.005', end_lr = '1e-5')