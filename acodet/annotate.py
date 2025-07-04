import time
from datetime import datetime as dt
from acodet import models
from acodet.funcs import (
    get_files,
    gen_annotations,
    get_dt_filename,
    remove_str_flags_from_predictions,
)
from acodet import global_config as conf
import pandas as pd
import numpy as np
from pathlib import Path


class MetaData:
    def __init__(self):
        """
        Initialize the MetaData class with the columns that will be used to
        store the metadata of the generated annotations.
        """
        self.filename = "filename"
        self.f_dt = "date from timestamp"
        self.n_pred_col = "number of predictions"
        self.avg_pred_col = "average prediction value"
        self.n_pred08_col = "number of predictions with thresh>0.8"
        self.n_pred09_col = "number of predictions with thresh>0.9"
        self.time_per_file = "computing time [s]"
        if not "timestamp_folder" in conf.session:
            self.df = pd.DataFrame(
                columns=[
                    self.filename,
                    self.f_dt,
                    self.n_pred_col,
                    self.avg_pred_col,
                    self.n_pred08_col,
                    self.n_pred09_col,
                ]
            )
        else:
            self.df = pd.read_csv(
                conf.session["timestamp_folder"].parent.parent.joinpath(
                    "stats.csv"
                )
            )
            self.df.pop("Unnamed: 0")

    def append_and_save_meta_file(
        self,
        file: Path,
        annot: pd.DataFrame,
        f_ind: int,
        timestamp_foldername: str,
        relativ_path: str = conf.SOUND_FILES_SOURCE,
        computing_time: str = "not calculated",
        **kwargs,
    ):
        """
        Append the metadata of the generated annotations to the dataframe and
        save it to a csv file.

        Parameters
        ----------
        file : Path
            Path to the file that was annotated.
        annot : pd.DataFrame
            Dataframe containing the annotations.
        f_ind : int
            Index of the file.
        timestamp_foldername : str
            Timestamp of the annotation run for folder name.
        relativ_path : str, optional
            Path of folder containing files , by default conf.SOUND_FILES_SOURCE
        computing_time : str, optional
            Amount of time that prediction took, by default "not calculated"
        """
        self.df.loc[f_ind, self.f_dt] = str(get_dt_filename(file).date())
        self.df.loc[f_ind, self.filename] = Path(file).relative_to(
            relativ_path
        )
        # TODO relative_path muss noch dauerhaft geändert werden
        self.df.loc[f_ind, self.n_pred_col] = len(annot)
        df_clean = remove_str_flags_from_predictions(annot)
        self.df.loc[f_ind, self.avg_pred_col] = np.mean(
            df_clean[conf.ANNOTATION_COLUMN]
        )
        self.df.loc[f_ind, self.n_pred08_col] = len(
            df_clean.loc[df_clean[conf.ANNOTATION_COLUMN] > 0.8]
        )
        self.df.loc[f_ind, self.n_pred09_col] = len(
            df_clean.loc[df_clean[conf.ANNOTATION_COLUMN] > 0.9]
        )
        self.df.loc[f_ind, self.time_per_file] = computing_time
        self.df.to_csv(
            Path(conf.GEN_ANNOTS_DIR)
            .joinpath(timestamp_foldername)
            .joinpath("stats.csv")
        )


def run_annotation(train_date=None, **kwargs):
    files = get_files(location=conf.SOUND_FILES_SOURCE, search_str="**/*")
    if not "timestamp_folder" in conf.session:
        timestamp_foldername = dt.strftime(dt.now(), "%Y-%m-%d_%H-%M-%S")
        timestamp_foldername += conf.ANNOTS_TIMESTAMP_FOLDER
        mdf = MetaData()
        f_ind = 0

    else:
        timestamp_foldername = conf.session[
            "timestamp_folder"
        ].parent.parent.stem
        last_annotated_file = list(
            conf.session["timestamp_folder"].rglob("*.txt")
        )[-1]
        file_stems = [f.stem for f in files]
        file_idx = np.where(
            np.array(file_stems) == last_annotated_file.stem.split("_annot")[0]
        )[0][0]
        files = files[file_idx:]
        mdf = MetaData()
        f_ind = file_idx - 1

    if not train_date:
        model = models.init_model()
        mod_label = conf.MODEL_NAME
    else:
        df = pd.read_csv("../trainings/20221124_meta_trainings.csv")
        row = df.loc[df["training_date"] == train_date]
        model_name = row.Model.values[0]
        keras_mod_name = row.keras_mod_name.values[0]

        model = models.init_model(
            model_instance=model_name,
            checkpoint_dir=f"../trainings/{train_date}/unfreeze_no-TF",
            keras_mod_name=keras_mod_name,
        )
        mod_label = train_date

    if conf.STREAMLIT:
        import streamlit as st

        st.session_state.progbar1 = 0
    for i, file in enumerate(files):
        if file.is_dir():
            continue

        if conf.STREAMLIT:
            import streamlit as st

            st.session_state.progbar1 += 1
        f_ind += 1
        start = time.time()
        annot = gen_annotations(
            file,
            model,
            mod_label=mod_label,
            timestamp_foldername=timestamp_foldername,
            num_of_files=len(files),
            **kwargs,
        )
        computing_time = time.time() - start
        mdf.append_and_save_meta_file(
            file,
            annot,
            f_ind,
            timestamp_foldername,
            computing_time=computing_time,
            **kwargs,
        )
    return timestamp_foldername


def check_for_multiple_time_dirs_error(path):
    if not path.joinpath(conf.THRESH_LABEL).exists():
        subdirs = [l for l in path.iterdir() if l.is_dir()]
        path = path.joinpath(subdirs[-1].stem)
    return path


def filter_annots_by_thresh(time_dir=None, **kwargs):
    if not time_dir:
        path = Path(conf.GEN_ANNOT_SRC)
    else:
        path = Path(conf.GEN_ANNOTS_DIR).joinpath(time_dir)
    files = get_files(location=path, search_str="**/*txt")
    files = [f for f in files if conf.THRESH_LABEL in str(f.parent)]
    path = check_for_multiple_time_dirs_error(path)
    for i, file in enumerate(files):
        try:
            annot = pd.read_csv(file, sep="\t")
        except Exception as e:
            print(
                "Could not process file, maybe not an annotation file?",
                "Error: ",
                e,
            )
        annot = annot.loc[annot[conf.ANNOTATION_COLUMN] >= conf.THRESH]
        save_dir = (
            path.joinpath(f"thresh_{conf.THRESH}")
            .joinpath(file.relative_to(path.joinpath(conf.THRESH_LABEL)))
            .parent
        )
        save_dir.mkdir(exist_ok=True, parents=True)

        if "Selection" in annot.columns:
            annot = annot.set_index('Selection')
        else:
            annot.index = np.arange(1, len(annot) + 1)
            annot.index.name = "Selection"
        if len(annot) > 0:
            try:
                check_selection_starts_at_1(annot)
            except AssertionError as e:
                print(e)
                vals = annot.index.values
                vals += 1
                annot.index = vals
                annot.index.name = "Selection"
        annot.to_csv(save_dir.joinpath(file.stem + file.suffix), sep="\t")
        if conf.STREAMLIT and "progbar1" in kwargs.keys():
            kwargs["progbar1"].progress((i + 1) / len(files), text="Progress")
        else:
            print(f"Writing file {i+1}/{len(files)}")
    if conf.STREAMLIT:
        return path

def check_selection_starts_at_1(annot):
    """
    Ensure that the index of the annotation DataFrame starts at 1.
    
    Parameters
    ----------
    annot : pd.DataFrame
        The annotation DataFrame to modify.
        
    Raises
    AssertionError
        If the index of the annotation DataFrame does not start at 1.
    """
    assert (annot.index.values[0] > 0), (
            "Annotation index needs to start above 0 to work with Raven."
            )
    

if __name__ == "__main__":
    train_dates = ["2022-11-30_01"]

    for train_date in train_dates:
        start = time.time()
        run_annotation(train_date)
        end = time.time()
        print(end - start)
