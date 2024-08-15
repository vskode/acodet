import numpy as np
import pandas as pd
from pathlib import Path
from hbdet.funcs import get_dt_filename


def create_day_dir(file_path):
    file_path.parent.joinpath(f.stem).mkdir(exist_ok=True)


main_path = Path("../Annais/Blue_whales/Annotations_bluewhales")

files = list(main_path.rglob("2*.txt"))
counter = 0
for f in files:
    data = pd.read_csv(f, sep="\t")
    data = data[data["Comments"] == "S"]
    if not "Begin File" in data.columns:
        continue
    else:
        create_day_dir(f)
        for beg_f in np.unique(data["Begin File"]):
            hour_data = data[data["Begin File"] == beg_f]
            hour = get_dt_filename(beg_f).hour
            new_data = hour_data.copy()
            new_data["Begin Time (s)"] -= hour * 1500
            new_data["End Time (s)"] -= hour * 1500
            save_str = f.parent.joinpath(f.stem).joinpath(
                Path(beg_f).stem + "_annotated.txt"
            )
            new_data.to_csv(save_str, sep="\t")
            counter += 1
            print(
                "most recent file:",
                f.stem,
                "files written: ",
                counter,
                end="\r",
            )
