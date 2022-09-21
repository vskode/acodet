import numpy as np
import matplotlib.pyplot as plt
import json
import time

def plot_model_results(unfreezes, path):

    fig, ax = plt.subplots(ncols = 4, nrows = 2, figsize = [15, 8])

    for unfreeze in unfreezes:
        checkpoint_path = f"{path}/unfreeze_{unfreeze}"

        f"{checkpoint_path}/results.json"

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
    fig.savefig(f'{path}/model_results_{ref_time}.png')

