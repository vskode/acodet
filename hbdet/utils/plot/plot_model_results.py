import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path


def plot_model_results(datetime):

    fig, ax = plt.subplots(ncols = 4, nrows = 2, figsize = [15, 8])

    checkpoint_paths = Path(f"trainings/{datetime}").glob('unfreeze_*')
    for checkpoint_path in checkpoint_paths:
        unfreeze = int(checkpoint_path.stem.split('_')[-1])

        with open(f"{checkpoint_path}/results.json", 'r') as f:
            results = json.load(f)

        for i, m in enumerate(results.keys()):
            row = i // 4
            col = np.mod(i, 4)
            # if row == 1 and col == 0:
            #     ax[row, col].plot(np.log10(results[m]), 
            #                     label = f'{unfreeze}')
            #     ax[row, col].set_title(r'log$_{10}$(val_loss)')
            # else:
            ax[row, col].plot(results[m], 
                            label = f'{unfreeze}')
            if row == 0:
                ax[row, col].set_title(f'{m}')
            if row == col == 0:
                ax[row, col].set_ylabel('training')
            elif row == 1 and col == 0:
                ax[row, col].set_ylabel('val')
    ax[0, 0].legend()

    today = time.ctime()
    fig.suptitle('Model Results '
                'with varying number of unfrozen layers\n'
                f'{today}')
    ref_time = time.strftime('%Y%m%d', time.gmtime())
    fig.savefig(f'trainings/{datetime}/model_results_{ref_time}.png')

if __name__ == '__main__':
    plot_model_results('2022-09-16_17')