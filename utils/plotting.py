#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
from pathlib import Path

#%%
if not 'Daten' in os.listdir():
    os.chdir('../..')
    
file_path = 'google_model_evaluation_v1.csv'
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
    if index < 3:
        dates.append( pd.to_datetime(Path(row.file).stem.split('.')[1], 
                                     format='%y%m%d%H%M%S') )
    else:
        dates.append( pd.to_datetime(Path(row.file).stem, 
                                     format='PAM_%Y%m%d_%H%M%S_000') )
    colors.append(c(row.quality_of_recording))

df['dates'] = dates
df['dates'] = pd.to_datetime(df['dates']).dt.date

# %%

fig, ax = plt.subplots(figsize = [14, 9])

ax.set_ylabel('MSE of predictions [%]')
ax.set_title('Stanton Bank prediction accuracy | model = google'\
                '\nnumber of annotations on bars')
ax.set_xticks(np.arange(len(df)))
ax.set_xticklabels(labels=df['dates'], rotation = 'vertical')

pps = ax.bar(np.arange(len(df)), df['mse(google)']*100, 
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

fig.savefig('google_pred_acc.png', dpi = 300, facecolor = 'white')
# %%
