import numpy as np
import matplotlib.pyplot as plt
import json
import time

unfreezes = [2, 4, 5, 7, 9, 10, 15, 20, 25]

fig, ax = plt.subplots(ncols = 4, nrows = 2, figsize = [15, 8])

for unfreeze in unfreezes:
    checkpoint_path = f"trainings/unfreeze_{unfreeze}_lr_exp"

    f"{checkpoint_path}/results.json"

    with open(f"{checkpoint_path}/results.json", 'r') as f:
        results = json.load(f)

    for i, m in enumerate(results.keys()):
        row = i // 4
        col = np.mod(i, 4)
        if row == 1 and col == 0:
            ax[row, col].plot(np.log10(results[m]), 
                            label = f'{unfreeze}')
            ax[row, col].set_title(r'log$_{10}$(val_loss)')
        else:
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
fig.suptitle('Model Results over 40 Epochs '
            'with varying number of unfrozen layers\n'
            f'{today}')
ref_time = time.strftime('%Y%m%d', time.gmtime())
fig.savefig(f'model_results_{ref_time}.png')

