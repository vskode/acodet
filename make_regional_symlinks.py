"""
A script to parse regional information about our various datasets and replicate the data in regional subdirectories.
The regional splits are created by symlinking to the original files so we don't need to duplicate the dataset.
The script expects to find a "data" dir in the current directory (which could be a symlink to the actual data/ dir).
"""
from pathlib import Path

import pandas as pd

#########################
## Parsing Ilaria's Data
#########################

# NOTE: There is one file missing from the training sheet: 5014.210214140002_annot_Humpback_20221130allnoise.txt
#       We're okay with this.
cr_train_df = pd.read_excel(Path("data/costa-rica-humpbacks/summary_predictions_per_file.xlsx"), sheet_name="training")
cr_test_df = pd.read_excel(Path("data/costa-rica-humpbacks/summary_predictions_per_file.xlsx"), sheet_name="testing")
country_map = {"PA": "Panama", "CR": "Costa_Rica"}


def make_cr_symlinks(df, split):
    root_dir = Path("data/costa-rica-humpbacks").resolve()
    tables_dir = root_dir / split / "tables"
    regional_tables_dir = root_dir / "regional" / split / "tables"
    regional_audio_dir = root_dir / "regional" / split / "recs"
    if not regional_audio_dir.is_symlink():
        regional_audio_dir.parent.mkdir(parents=True, exist_ok=True)
        regional_audio_dir.symlink_to(tables_dir.parent / "recs")

    regions = set()
    for idx, row in df.iterrows():
        source = tables_dir / row["file"]
        dest = regional_tables_dir / country_map[row["Country"]] / row["location"] / row["file"]
        regions.add(str(dest.parent))
        if not dest.is_symlink():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(source)

    print(f"Regions in {split} split:")
    for r in regions:
        print("    " + r)


make_cr_symlinks(cr_train_df, "training")
make_cr_symlinks(cr_test_df, "testing")

##########################
## Parsing Vincent's Data
##########################

info_df = pd.read_csv(Path("data/vkather-humpbacks/Dataset_Information.csv"))
region_map = {}
for idx, row in info_df.iterrows():
    region_map[row["Dataset name"]] = row["Region"]

tables_dir = Path("data/vkather-humpbacks/tables").resolve()
regional_tables_dir = Path("data/vkather-humpbacks/regional/tables").resolve()
regional_tables_dir.mkdir(parents=True, exist_ok=True)
# regional_audio_dir = Path("data/vkather-humpbacks/regional/audio").resolve()
# if not regional_audio_dir.is_symlink():
#     regional_audio_dir.symlink_to(tables_dir.parent / "audio")

regions = set()
for region, group in info_df.groupby("Region"):
    for subfolder in group["Dataset name"]:
        source = tables_dir / subfolder
        dest = regional_tables_dir / region / subfolder
        regions.add(str(dest.parent))
        if not dest.is_symlink():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(source)

print(f"Regions in vkather test data:")
for r in regions:
    print("    " + r)
