import json
from pathlib import Path 
import global_config as conf
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('../Data/Datasets_metadata.csv')

TFRECORDS_DIR = list(Path(conf.TFREC_DESTINATION).iterdir())
stems = [f.stem for f in TFRECORDS_DIR]
stems_in_df = [s for s in stems if s in df['name_on_vincents_harddrive'].values]
df.index = df['name_on_vincents_harddrive']
df = df.loc[stems_in_df]

locs = pd.DataFrame(columns=['lat', 'long'])
for stem in stems_in_df:
    locs.loc[stem, :] = (df.loc[stem, 'Lat'], df.loc[stem, 'Long'])
locs.loc['SABA01_201511_201604_SN275', :] = (17.5116, -63.1932)
locs.loc['SABA01_201604_201608_SN276', :] = (17.5116, -63.1932)
locs.loc['CHALLENGER_AMAR123', :] = (32.2928362337, -64.7829419359)
locs.loc['SAMOSAS_EL1_2021', :] = (57.09847, -8.968883)
locs.loc['SAMOSAS_S1_2021', :] = (56.53263, -8.856400)
locs.loc['SAMOSAS_N1_2021', :] = (58.09180, -8.913433)
locs.loc['StantonBank_2kHz_D5_2019', :] = (56.253869, -8.063398)
locs.loc['StantonBank_2kHz_D2_2018', :] = (56.253869, -8.063398)
locs.loc['ScotWest_v5_2khz', :] = (56.253869, -8.063398)
locs.loc['ScotWest_v4_2khz', :] = (56.253869, -8.063398)
locs.loc['Tolsta_2kHz_D2_2018', :] = (58.330404, -5.962842)
locs.loc['HWDT_JOIN_2019', :] = (57.172271, -6.907990)
locs.loc['SALLY_TUCKERS_AMAR088.1', :] = (32.551245, -64.579762)


for path in TFRECORDS_DIR:
    locs.loc[path.stem, 'noise'] = 0
    locs.loc[path.stem, 'calls'] = 0
    for file in path.glob('**/*json'):
        res = json.load(open(file))
        locs.loc[path.stem, 'noise'] += sum(res['dataset']['noise'].values())
        locs.loc[path.stem, 'calls'] += sum(res['dataset']['calls'].values())
        
locs = locs.fillna(0)
locs['dirs'] = locs.iloc[:, 0].index.values
locs.index = range(len(locs))
locs.loc[((locs.lat > 50) * (locs.lat < 59)), 'Region'] = 'Scotland'
locs.loc[(locs.lat >= 59), 'Region'] = 'Iceland'
locs.loc[((locs.lat < 50) * (locs.lat > 39)), 'Region'] = 'US_Bermuda'
locs.loc[(locs.lat < 39), 'Region'] = 'Caribbean'
locs.to_csv('../Data/location_df.csv')

regions = dict()
regions['Scotland'] = locs[locs.lat > 50]
regions['Scotland'] = (sum(regions['Scotland'][regions['Scotland'].lat < 59]['calls'].values),
                       sum(regions['Scotland'][regions['Scotland'].lat < 59]['noise'].values))
regions['Iceland'] = (sum(locs[locs.lat >= 59]['calls'].values),
                       sum(locs[locs.lat >= 59]['noise'].values))
regions['US_Bermuda'] = locs[locs.lat < 50]
regions['US_Bermuda'] = (sum(regions['US_Bermuda'][regions['US_Bermuda'].lat > 39]['calls'].values),
                       sum(regions['US_Bermuda'][regions['US_Bermuda'].lat > 39]['noise'].values))
regions['Caribbean'] = (sum(locs[locs.lat < 39]['calls'].values),
                       sum(locs[locs.lat < 39]['noise'].values))


def plot_sized_bubbles(locs):
    plt.figure(figsize=[12, 10])
    map = Basemap(width=8000000,
                height=6000000,
                projection='lcc',
                resolution=None,
                lat_0=42, lat_1=40, lat_2=50,
                lon_0=-41)
    map.shadedrelief()

    for i, row in locs.iterrows():
        loc_x, loc_y = map(row.long, row.lat)
        size = sum([row.noise, row.calls])/20568*5000
        
        map.scatter(loc_x, loc_y, marker='o', s=size, alpha=0.7)
    # plt.show()
    plt.tight_layout()
    plt.savefig('../Data/Dataset_map.png', dpi = 200)
    
    
def plot_pies(locs):
    plt.figure()
    map = Basemap(width=8000000,
                height=6000000,
                projection='lcc',
                resolution=None,
                lat_0=42, lat_1=40, lat_2=50,
                lon_0=-41)
    map.shadedrelief()
    c = ('tab:blue', 'tab:orange')

    for i, row in locs.iterrows():
        loc_x, loc_y = map(row.long, row.lat)
        cumsum = np.cumsum([row.noise, row.calls])
        cumsum = cumsum/ (cumsum[-1]+1)
        pie = [0] + cumsum.tolist()

        for i, (r1, r2) in enumerate(zip(pie[:-1], pie[1:])):
            angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()

            xy = np.column_stack([x, y])
            marker = xy
            size = sum([row.noise, row.calls])/20568*5000
            
            map.scatter(loc_x, loc_y, marker=marker, s=size, alpha=0.7, color=c[i])
    # map.legend()
    # plt.show()
    plt.tight_layout()
    plt.savefig('../Data/Dataset_map_pies.png', dpi = 200)
    # return ax
    
plot_sized_bubbles(locs)