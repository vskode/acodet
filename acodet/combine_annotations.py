import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import glob
from pathlib import Path
from acodet.funcs import remove_str_flags_from_predictions, get_files
import os
import numpy as np
import acodet.global_config as conf
import json
from tqdm import tqdm

with open("acodet/annotation_mappers.json", "r") as m:
    mappers = json.load(m)


# TODO aufraeumen
# annotation_files = Path(r'/mnt/f/Daten/20221019-Benoit/').glob('**/*.txt')
# annotation_files = Path(r'generated_annotations/2022-11-04_12/').glob('ch*.txt')


def compensate_for_naming_inconsistencies(hard_drive_path, file):
    split_file = file.stem.split("Table")[0]
    file_path = glob.glob(
        f"{hard_drive_path}/**/{split_file}*wav", recursive=True
    )

    if not file_path:
        file_tolsta = "336097327." + split_file[6:].replace(
            "_000", ""
        ).replace("_", "")
        file_path = glob.glob(
            f"{hard_drive_path}/**/{file_tolsta}*wav", recursive=True
        )

    if not file_path:
        file_tolsta = "335564853." + split_file[6:].replace(
            "5_000", "4"
        ).replace("_", "")
        file_path = glob.glob(
            f"{hard_drive_path}/**/{file_tolsta}*wav", recursive=True
        )

    if not file_path:
        file_new_annot = file.stem.split("_annot")[0]
        file_path = glob.glob(
            f"{hard_drive_path}/**/{file_new_annot}*", recursive=True
        )

    if not file_path:
        split_file_underscore = file.stem.split("_")[0]
        file_path = glob.glob(
            f"{hard_drive_path}/**/{file_new_annot}*wav", recursive=True
        )
        if not file_path:
            file_new_annot = split_file_underscore.split(".")[-1]
            file_path = glob.glob(
                f"{hard_drive_path}/**/*{file_new_annot}*wav", recursive=True
            )

    if not file_path:
        print(f"Unable to find sound file for {file.stem} under sound directory {hard_drive_path}.")
        return False
    return file_path


def get_corresponding_sound_file(file):
    hard_drive_path = conf.SOUND_FILES_SOURCE
    file_path = glob.glob(
        f"{hard_drive_path}/**/{file.stem}*wav", recursive=True
    )
    if not file_path:
        file_path = compensate_for_naming_inconsistencies(
            hard_drive_path, file
        )

    if not file_path:
        # TODO fehler raisen, dass bringt so einfach nichts
        return "empty"

    if len(file_path) > 1:
        p_dir = list(file.relative_to(conf.REV_ANNOT_SRC).parents)[-2]
        p_dir_main = str(p_dir).split("_")[0]
        for path in file_path:
            if p_dir_main in path:
                file_path = path
    else:
        file_path = file_path[0]

    if isinstance(file_path, list) and len(file_path) > 1:
        file_path = file_path[0]
        print(
            "WARNING: Multiple sound files for annotations file found."
            " Because pattern could not be resolved, first file is chosen."
            f"\nannotations file name: \n{file}\n"
            f"sound file name: \n{file_path}\n"
        )

    return file_path


def seperate_long_annotations(df):
    bool_long_annot = df["End Time (s)"] - df["Begin Time (s)"] > round(
        conf.CONTEXT_WIN / conf.SR
    )
    for i, row in df.loc[bool_long_annot].iterrows():
        n_new_annots = int(
            (row["End Time (s)"] - row["Begin Time (s)"])
            / (conf.CONTEXT_WIN / conf.SR)
        )
        begins = row["Begin Time (s)"] + np.arange(n_new_annots) * (
            conf.CONTEXT_WIN / conf.SR
        )
        ends = begins + (conf.CONTEXT_WIN / conf.SR)
        n_df = pd.DataFrame()
        for col in row.keys():
            n_df[col] = [row[col]] * n_new_annots
        n_df["Begin Time (s)"] = begins
        n_df["End Time (s)"] = ends
        n_df["Selection"] = np.arange(n_new_annots) + row["Selection"]
        df = pd.concat(
            [df.drop(row.name), n_df]
        )  # delete long annotation from df
    return df


def label_explicit_noise(df):
    # This is a noise file, so everything that got a numeric prediction score of > 0.9 should be
    # considered a "hard negative" (a.k.a. explicit noise).
    expl_noise_crit_idx = np.where(pd.to_numeric(df[conf.ANNOTATION_COLUMN], errors="coerce") > 0.9)[0]
    df.loc[expl_noise_crit_idx, "label"] = -1
    return df


def differentiate_label_flags(df, flag=None):
    # df.loc[:, conf.ANNOTATION_COLUMN].fillna(value="c", inplace=True)
    if flag == "calls":
        if conf.ANNOTATION_COLUMN not in df.columns:
            df.loc[:, conf.ANNOTATION_COLUMN] = 'c'
    elif flag == "noise":
        if conf.ANNOTATION_COLUMN not in df.columns:
            df.loc[:, conf.ANNOTATION_COLUMN] = 'n'
    if isinstance(df[conf.ANNOTATION_COLUMN][0], str):
        df.loc[df[conf.ANNOTATION_COLUMN].str.strip().str.lower() == "c", "label"] = 1
        df.loc[df[conf.ANNOTATION_COLUMN].str.strip().str.lower() == "n", "label"] = -1

    df_std = seperate_long_annotations(df)

    if isinstance(df[conf.ANNOTATION_COLUMN][0], str):
        df_std = df_std.drop(
            df_std.loc[df_std[conf.ANNOTATION_COLUMN].str.strip().str.lower() == "u"].index
        )
        
    df_std.index = pd.RangeIndex(0, len(df_std))
    if flag == "noise":
        df_std = label_explicit_noise(df_std)

    return df_std


def get_labels(file, df, active_learning=False, inbetween_noise=False, clip_duration=3.8775):
    if not active_learning:
        df["label"] = 1
    else:
        noise_flag, annotated_flag, calls_flag = [
            "_allnoise",
            "_annotated",
            "_allcalls",
        ]
        df = df.iloc[df.Selection.drop_duplicates().index]
        if calls_flag in file.stem:
            df["label"] = 1
            df = differentiate_label_flags(df, flag="calls")
        elif noise_flag in file.stem:
            df["label"] = 0
            df = differentiate_label_flags(df, flag="noise")
        elif annotated_flag in file.stem:
            if conf.ANNOTATION_COLUMN in df.columns:
                df_clean = remove_str_flags_from_predictions(df)
                df.loc[df_clean.index, conf.ANNOTATION_COLUMN] = "u"
                df = differentiate_label_flags(df)
            else:
                df = differentiate_label_flags(df, flag="calls")
                
    if inbetween_noise:
        df = df.sort_values("Begin Time (s)")
        # The 1e-5 is padding to avoid missing gaps due to rounding error.
        between_annotations_durations = df["Begin Time (s)"][1:].values - df["End Time (s)"][:-1].values + 1e-5
        gap_rows = (between_annotations_durations >= clip_duration)
        prev_rows = df.iloc[:-1][gap_rows]
        new_rows = df.to_dict("records")
        for (idx, prev_row), duration in zip(prev_rows.iterrows(), between_annotations_durations[gap_rows]):
            row = prev_row.to_dict()
            for i in range(int(duration // clip_duration)):
                row["Begin Time (s)"] = row["End Time (s)"]
                row["End Time (s)"] = row["Begin Time (s)"] + clip_duration
                row["label"] = 0
                row[conf.ANNOTATION_COLUMN] = "implicit noise"
                row["Selection"] = len(new_rows)
                new_rows.append(row.copy())
        df = pd.DataFrame.from_records(new_rows, columns=df.columns).sort_values("Begin Time (s)")
    return df


def standardize(
    df, *, mapper, filename_col="filename", selection_col="Selection"
):
    keep_cols = ["label", "start", "end", "freq_min", "freq_max"]
    df = df.rename(columns=mapper)
    if not "end" in df.columns:
        df["end"] = df.start + (df["End Time (s)"] - df["Begin Time (s)"])
    out_df = df[keep_cols]
    out_df.index = pd.MultiIndex.from_arrays(
        arrays=(df[filename_col], df[selection_col])
    )
    return out_df.astype(dtype=np.float64)


def filter_out_high_freq_and_high_transient(df):
    df = df.loc[df["High Freq (Hz)"] <= 2000]
    df = df.loc[df["End Time (s)"] - df["Begin Time (s)"] >= 0.4]
    return df


def finalize_annotation(file, freq_time_crit=False, **kwargs):
    try:
        ann = pd.read_csv(file, sep="\t")
    except pd.errors.EmptyDataError as e:
        print(f"Empty file {file}: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if ann.empty:
        print(f"Empty dataframe from file {file}")
        return pd.DataFrame(), pd.DataFrame()

    ann["filename"] = get_corresponding_sound_file(file)
    # if not ann['filename']:
    #     print(f'corresponding sound file for annotations file: {file} not found')

    ann = get_labels(file, ann, **kwargs)
    if "File Offset (s)" in ann.columns:
        mapper = mappers["file_offset_mapper"]
    else:
        mapper = mappers["default_mapper"]

    if freq_time_crit:
        ann = filter_out_high_freq_and_high_transient(ann)

    ann_explicit_noise = ann.loc[ann["label"] == -1, :]
    ann_explicit_noise["label"] = 0
    ann = ann.drop(ann.loc[ann["label"] == -1].index)
    std_annot_train = standardize(ann, mapper=mapper)
    std_annot_enoise = standardize(ann_explicit_noise, mapper=mapper)

    return std_annot_train, std_annot_enoise


def leading_underscore_in_parent_dirs(file):
    return "_" in [f.stem[0] for f in list(file.parents)[:-1]]


def get_active_learning_files(files):
    cases = ["_allnoise", "_annotated", "_allcalls"]
    cleaned_files = [f for f in files if [True for c in cases if c in f.stem]]
    drop_cases = ["_tobechecked"]
    final_cleanup = [
        f
        for f in cleaned_files
        if not [True for d in drop_cases if d in f.stem]
    ]
    return final_cleanup


def generate_final_annotations(
    annotation_files=None, active_learning=True, **kwargs
):
    if not annotation_files:
        annotation_files = get_files(
            location=conf.REV_ANNOT_SRC, search_str="**/*.txt"
        )
    files = list(annotation_files)
    if active_learning:
        files = get_active_learning_files(files)
        print(f"Filtered {len(annotation_files)} annotated files down to {len(files)} files for active learning.")
    files.sort()
    if not files:
        raise ValueError(f"No .txt files found under {conf.REV_ANNOT_SRC}")

    df_t, df_n = pd.DataFrame(), pd.DataFrame()
    for file in tqdm(files, 'Gathering annotations from files', len(files)):
        if leading_underscore_in_parent_dirs(file):
            print(
                file,
                " skipped due to leading underscore in parent dir.",
            )
            continue
        df_train, df_enoise = finalize_annotation(
            file,
            active_learning=active_learning,
            **kwargs,
        )
        df_t = pd.concat([df_t, df_train])
        df_n = pd.concat([df_n, df_enoise])

    total_pos = (df_t["label"] == 1).sum() + (df_n["label"] == 1).sum()
    total_neg = (df_t["label"] == 0).sum() + (df_n["label"] == 0).sum()
    assert (len(df_t) + len(df_n)) == total_pos + total_neg
    print(f"Total annotations: {total_pos + total_neg} ({total_pos} positive, {total_neg} negative)")

    # Assign to test set automatically if the name indicates it.
    if 'test' in conf.REV_ANNOT_SRC:
        df_t.loc[:, 'subset'] = 'test'
        df_n.loc[:, 'subset'] = 'test'

    save_dir = Path(conf.ANNOT_DEST)
    save_dir.mkdir(exist_ok=True, parents=True)
    df_t.to_csv(save_dir.joinpath("combined_annotations.csv"))
    df_n.to_csv(save_dir.joinpath("explicit_noise.csv"))
    print(f"Annotations saved to {save_dir}")

    # If this is a train set, mark the train/validation split, if provided.
    if 'train' in conf.REV_ANNOT_SRC:
        add_validation_labels(save_dir)


def add_validation_labels(save_dir):
    """
    Use validation_files.csv to assign what's val and what's train.
    This is to ensure that we are using the same validation set.
    """
    val_file = Path(conf.REV_ANNOT_SRC).resolve() / "validation_files.csv"
    if not val_file.is_file():
        print("No validation_files.csv file found. Cannot assign train/validation subset labels.")
        return

    df_val = pd.read_csv(val_file, dtype=str)
    df_t = pd.read_csv(save_dir.joinpath("combined_annotations.csv"))
    df_n = pd.read_csv(save_dir.joinpath("explicit_noise.csv"))
    df_t["subset"] = "train"
    df_n["subset"] = "train"
    stems_t = np.array([Path(f).stem for f in df_t["filename"]])
    stems_n = np.array([Path(f).stem for f in df_n["filename"]])
    for file in df_val["validation_files"].values:
        bool_stems_t = np.char.startswith(stems_t, str(file))
        bool_stems_n = np.char.startswith(stems_n, str(file))
        df_t.loc[bool_stems_t, "subset"] = "val"
        df_n.loc[bool_stems_n, "subset"] = "val"

    total_train = (df_t["subset"] == "train").sum() + (df_n["subset"] == "train").sum()
    total_val = (df_t["subset"] == "val").sum() + (df_n["subset"] == "val").sum()
    assert (len(df_t) + len(df_n)) == total_train + total_val
    print(f"Assigned {total_train} to training, {total_val} to validation.")

    df_t.to_csv(save_dir.joinpath("combined_annotations.csv"), index=False)
    df_n.to_csv(save_dir.joinpath("explicit_noise.csv"), index=False)


if __name__ == "__main__":
    annotation_files = list(Path(conf.REV_ANNOT_SRC).glob("**/*.txt"))
    if len(annotation_files) == 0:
        annotation_files = list(Path(conf.REV_ANNOT_SRC).glob("*.txt"))
    generate_final_annotations(
        annotation_files, active_learning=True, freq_time_crit=False
    )
