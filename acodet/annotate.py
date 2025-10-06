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
from tqdm import tqdm

class MetaData:
    def __init__(self, timestamp_foldername: str):
        """
        Initialize the MetaData class with the columns that will be used to
        store the metadata of the generated annotations.
        
        timestamp_foldername : str
            Timestamp of the annotation run for folder name.
        """
        self.save_dir = Path(conf.GEN_ANNOTS_DIR) / timestamp_foldername
        self.file_col = "filename"
        self.f_dt = "date from timestamp"
        self.n_pred_col = "number of predictions"
        self.avg_pred_col = "average prediction value"
        self.n_pred08_col = "number of predictions with thresh>0.8"
        self.n_pred09_col = "number of predictions with thresh>0.9"
        self.time_per_file = "computing time [s]"
        if not "timestamp_folder" in conf.session:
            self.df = pd.DataFrame(
                columns=[
                    self.file_col,
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
        relativ_path : str, optional
            Path of folder containing files , by default conf.SOUND_FILES_SOURCE
        computing_time : str, optional
            Amount of time that prediction took, by default "not calculated"
        """
        self.df.loc[f_ind, self.f_dt] = str(get_dt_filename(file).date())
        self.df.loc[f_ind, self.file_col] = Path(file).relative_to(
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
        self.df.to_csv(self.save_dir.joinpath("stats.csv"))
        
    
    def multi_class_metadata(self):
        from tqdm import tqdm
        
        multi_df = pd.DataFrame()
        thresh_exceeding_classes = [
            d.stem for d in (self.save_dir / 'thresh_0.5').iterdir() 
            if not d.stem in ['All_Combined', 'multiclass']
            ]
        label_dict = {}
        if conf.STREAMLIT:
            import streamlit as st
            prog1 = st.progress(0, text='Building multiclass metadata file')
        for idx, lab in enumerate(tqdm(thresh_exceeding_classes)):
            preds = []
            df_pred = 0
            files = [
                d for d in (self.save_dir / 'thresh_0.5').rglob(f'*{lab}*.txt')
                if not 'combined' in d.stem and not 'multiclass' in d.stem
                ]
            for f in files:
                df = pd.read_csv(f, sep='\t')
                if len(df) > df_pred:
                    most_active = str(f)
                preds.extend(df[conf.ANNOTATION_COLUMN].values.tolist())
            if preds:
                label_dict[lab] = dict()
                label_dict[lab]['all_preds'] = preds
                label_dict[lab]['most_active'] = most_active
            if conf.STREAMLIT:
                prog1.progress(idx / len(thresh_exceeding_classes))
        
        multi_df['labels'] = list(label_dict.keys())
        multi_df['avg_confidence'] = [np.mean(label_dict[l]['all_preds']) for l in label_dict.keys()]
        multi_df['std_confidence'] = [np.std(label_dict[l]['all_preds']) for l in label_dict.keys()]
        multi_df['labels_by_occurrence'] = [len(label_dict[l]['all_preds']) for l in label_dict.keys()]
        multi_df['most_active_file'] = [Path(label_dict[l]['most_active']).relative_to(self.save_dir) for l in label_dict.keys()]
        multi_df = multi_df.sort_values('labels_by_occurrence', ascending=False)
        multi_df.to_csv(self.save_dir.joinpath('mutliclass_df.csv'))
            

def run_annotation(train_date=None, **kwargs):
    files = get_files(location=conf.SOUND_FILES_SOURCE, search_str="*.[wW][aA][vV]")
    if not "timestamp_folder" in conf.session:
        timestamp_foldername = dt.strftime(dt.now(), "%Y-%m-%d_%H-%M-%S")
        timestamp_foldername += conf.ANNOTS_TIMESTAMP_FOLDER
        mdf = MetaData(timestamp_foldername)
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
        mdf = MetaData(timestamp_foldername=timestamp_foldername)
        
        f_ind = file_idx - 1

    if not train_date:
        model = models.init_model(timestamp_foldername=timestamp_foldername)
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
        if 'callbacks' in kwargs:
            st.session_state.progbar1 = st.progress(0, text='Current file')
                
            st.session_state.progbar2 = st.progress(0, text='Overall progress')
    for i, file in tqdm(enumerate(files),
                        "Annotating files",
                        total=len(files),
                        leave=False):
        if file.is_dir():
            continue

        if conf.STREAMLIT:
            import streamlit as st
            kwargs['progbar2'].progress(i / len(files), 
                                               text='Overall progress')
        f_ind += 1
        start = time.time()
        try:
            annot = gen_annotations(
                file,
                model,
                mod_label=mod_label,
                timestamp_foldername=timestamp_foldername,
                num_of_files=len(files),
                **kwargs,
            )
        except Exception as e:
            print('Annotations could not be generated due to error:', e)
            continue
        computing_time = time.time() - start
        mdf.append_and_save_meta_file(
            file,
            annot,
            f_ind,
            computing_time=computing_time,
            **kwargs,
        )
    mdf.multi_class_metadata()
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
    if path.parent == Path('.'):
        path = Path(conf.GEN_ANNOTS_DIR).joinpath(path)
        
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
        if 'multiclass' in file.stem:
            # preds, species = np.array([s.split('__') for s in annot[conf.ANNOTATION_COLUMN]])
            if len(annot) == 0:
                continue
            preds = np.array([s.split('__') for s in annot[conf.ANNOTATION_COLUMN]])[:, 0]
            preds = np.array(preds, dtype=np.float32)
            
            annot = annot.loc[preds >= conf.THRESH]
        elif not conf.ANNOTATION_COLUMN in annot.columns:
            if 'combined' in file.stem:
                label_columns = (
                    annot.columns[~annot.columns.isin([
                        'Selection', 
                        'Begin Time (s)', 'End Time (s)', 
                        'High Freq (Hz)', 'Low Freq (Hz)'
                        ])]
                )
                df_bool = pd.DataFrame()
                for col in label_columns:
                    df_bool[col] = annot[col] >= conf.THRESH
                bool_any = np.where(df_bool.values)[0]
                bool_any = np.unique(bool_any)
                annot = annot.iloc[bool_any]
        else:
            annot = annot.loc[annot[conf.ANNOTATION_COLUMN] >= conf.THRESH]
        if len(annot) == 0:
            continue
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
            print(f"Writing file {i+1}/{len(files)}", end='\r')
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
