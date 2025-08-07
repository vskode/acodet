import pandas as pd
import numpy as np
from acodet.funcs import get_files, get_dt_filename
import acodet.global_config as conf
from pathlib import Path
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

sns.set_theme()
sns.set_style("white")


def hourly_prs(df: pd.DataFrame, lim: int = 10):
    """
    Compute hourly presence.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing annotations
    lim : int, optional
        limit for binary presence judgement, by default 10

    Returns
    -------
    int
        either 0 or 1 - 0 if less than lim annotations are present, 1 if more
    """
    if len(df) > lim:
        return 1
    else:
        return 0


def daily_prs(df: pd.DataFrame):
    """
    Compute daily presence. If at least one hour is present, the day is
    considered present.

    Parameters
    ----------
    df : pd.Dataframe
        dataframe containing annotations

    Returns
    -------
    int
        0 or 1 - 0 if no hour is present, 1 if at least one hour is present
    """
    if 1 in df.loc[len(df), h_of_day_str()].values:
        return 1
    else:
        return 0


def get_val(path: Path):
    """
    Get validation dataframe.

    Parameters
    ----------
    path : str or Path
        path to validation dataframe

    Returns
    -------
    pd.Dataframe
        validation dataframe
    """
    return pd.read_csv(path)


def h_of_day_str():
    return ["%.2i:00" % i for i in np.arange(24)]


def find_thresh05_path_in_dir(time_dir):
    """
    Get corrects paths leading to thresh_0.5 directory contatining annotations.
    Correct for incorrect paths, that already contain the thresh_0.5 path.

    Parameters
    ----------
    time_dir : str
        if run.py is run directly, a path to a specific timestamp can be passed

    Returns
    -------
    pathlib.Path
        correct path leading to thresh_0.5 directory
    """
    root = Path(conf.GEN_ANNOT_SRC)
    if root.parts[-1] == conf.THRESH_LABEL:
        root = root.parent
    elif root.parts[-1] == "thresh_0.9":
        root = root.parent

    if not time_dir:
        if root.joinpath(conf.THRESH_LABEL).exists():
            path = root.joinpath(conf.THRESH_LABEL)
        else:
            path = root
    else:
        path = (
            Path(conf.GEN_ANNOTS_DIR)
            .joinpath(time_dir)
            .joinpath(conf.THRESH_LABEL)
        )
    return path


def init_date_tuple(files):
    dates = list(
        map(lambda x: get_dt_filename(x.stem.split("_annot")[0]), files)
    )
    date_hour_tuple = list(
        map(lambda x: (str(x.date()), "%.2i:00" % int(x.hour)), dates)
    )

    return np.unique(date_hour_tuple, axis=0, return_counts=True)


def compute_hourly_pres(
    time_dir=None,
    thresh=conf.THRESH,
    lim=conf.SIMPLE_LIMIT,
    thresh_sc=conf.SEQUENCE_THRESH,
    lim_sc=conf.SEQUENCE_LIMIT,
    sc=False,
    fetch_config_again=False,
    **kwargs,
):
    if fetch_config_again:
        import importlib

        importlib.reload(conf)
        thresh = conf.THRESH
        lim = conf.SIMPLE_LIMIT
        thresh_sc = conf.SEQUENCE_THRESH
        lim_sc = conf.SEQUENCE_LIMIT

    path = find_thresh05_path_in_dir(time_dir)

    if "multi_datasets" in conf.session:
        directories = [
            [d for d in p.iterdir() if d.is_dir()]
            for p in path.iterdir()
            if p.is_dir()
        ][0]
    else:
        directories = [p for p in path.iterdir() if p.is_dir()]
    directories = [d for d in directories if not d.stem == "analysis"]

    for ind, fold in enumerate(directories):
        files = get_files(location=fold, search_str="**/*txt")
        files.sort()

        annots = return_hourly_pres_df(
            files,
            thresh,
            thresh_sc,
            lim,
            lim_sc,
            sc,
            fold,
            dir_ind=ind,
            total_dirs=len(directories),
            **kwargs,
        )
        if "save_filtered_selection_tables" in kwargs:
            top_dir_path = path.parent.joinpath(conf.THRESH_LABEL).joinpath(
                fold.stem
            )
        else:
            top_dir_path = path.joinpath(fold.stem)

        annots.df.to_csv(get_path(top_dir_path, conf.HR_PRS_SL))
        annots.df_counts.to_csv(get_path(top_dir_path, conf.HR_CNTS_SL))
        if not "dont_save_plot" in kwargs.keys():
            for metric in (conf.HR_CNTS_SL, conf.HR_PRS_SL):
                plot_hp(top_dir_path, lim, thresh, metric)

        if sc:
            annots.df_sc.to_csv(get_path(top_dir_path, conf.HR_PRS_SC))
            annots.df_sc_cnt.to_csv(get_path(top_dir_path, conf.HR_CNTS_SC))
            if not "dont_save_plot" in kwargs.keys():
                for metric in (conf.HR_CNTS_SC, conf.HR_PRS_SC):
                    plot_hp(top_dir_path, lim_sc, thresh_sc, metric)
        print("\n")


def get_end_of_last_annotation(annotations):
    """
    Get number of seconds from beginning to the end of the last annotation.

    Parameters
    ----------
    annotations : pd.DataFrame
        annotation dataframe

    Returns
    -------
    int or bool
        False or number of seconds until last annotation
    """
    if len(annotations) == 0:
        return False
    else:
        return int(annotations["End Time (s)"].iloc[-1])


def init_new_dt_if_exceeding_3600_s(h, date, hour):
    """
    Return new date and hour string if annotations exceed an hour. This
    ensures that hour presence is still computed even if a recording
    exceeds an hour.

    Parameters
    ----------
    h : int
        number of hours
    date : str
        date string
    hour : str
        hour string

    Returns
    -------
    tuple
        date and hour string
    """
    if h > 0:
        new_dt = dt.datetime.strptime(
            date + hour, "%Y-%m-%d%H:00"
        ) + dt.timedelta(hours=1)
        date = str(new_dt.date())
        hour = "%.2i:00" % new_dt.hour
    return date, hour


class ProcessLimits:
    def __init__(
        self,
        files,
        thresh,
        thresh_sc,
        lim,
        lim_sc,
        sc,
        dir_ind,
        total_dirs,
        return_counts,
    ):
        """
        Handle processing of hourly and daily annotations counts and presence.
        A class object is created and used to go through all files and calculate
        the presence and annotation count metrics for a given hour within the
        file. If more than one hour exists within a file, the metrics are counted
        for every hour. If multiple files make up one hour they are concatenated
        before processing. Simple limit and sequence limit processing is handled
        by this class.

        Parameters
        ----------
        files : list
            pathlib.Path objects linking to file path
        thresh : float
            threshold
        thresh_sc : float
            threshold for sequence limit
        lim : int
            limit for simple limit presence
        lim_sc : int
            limit for sequence limit
        sc : bool
            sequence limit yes or no
        dir_ind : int
            directory index, in case multiple dirs are processed
        total_dirs : int
            number of total directories
        return_counts : bool
            return annotation counts or only binary presence
        """
        self.df = pd.DataFrame(
            columns=["Date", conf.HR_DP_COL, *h_of_day_str()]
        )
        self.df_sc = self.df.copy()
        self.df_counts = pd.DataFrame(
            columns=["Date", conf.HR_DA_COL, *h_of_day_str()]
        )
        self.df_sc_cnt = self.df_counts.copy()
        self.files = files
        self.thresh = thresh
        self.thresh_sc = thresh_sc
        self.sc = sc
        self.lim_sc = lim_sc
        self.lim = lim
        self.dir_ind = dir_ind
        self.total_dirs = total_dirs
        self.return_counts = return_counts

        self.file_ind = 0
        self.row = 0
        self.n_prec_preds = conf.SEQUENCE_CON_WIN
        self.n_exceed_thresh = conf.SEQUENCE_LIMIT

    def concat_files_within_hour(self, count):
        """
        Concatenate files within one hour. Relevant if multiple files make
        up one hour.

        Parameters
        ----------
        count : int
            number of files making up given hour
        """
        self.annot_all = pd.DataFrame()
        self.filtered_annots = pd.DataFrame()
        for _ in range(count):
            self.annot_all = pd.concat(
                [
                    self.annot_all,
                    pd.read_csv(self.files[self.file_ind], sep="\t"),
                ]
            )
            self.file_ind += 1

    def seq_crit(self, annot):
        """
        Sequence limit calculation. Initially all predictions are thresholded.
        After that two boolean arrays are created. Using the AND operator for
        these two arrays only returns the values that pass the sequence limit.
        This means that within the number of consecutive windows
        (self.n_prec_anns) more than the self.lim_sc number of windows have to
        exceed the value of self.thresh_sc. Depending on the settings this
        function is cancelled early if only binary presence is of interest.
        A filtered annotations dataframe is saved that only has the predictions
        that passed the filtering process.

        Parameters
        ----------
        annot : pandas.DataFrame
            annotations of the given hour

        Returns
        -------
        int
            either 1 if only binary presence is relevant or the number of
            annotations that passed the sequence limit in the given hour
        """
        annot = annot.loc[annot[conf.ANNOTATION_COLUMN] >= self.thresh_sc]
        for i, row in annot.iterrows():
            bool1 = 0 <= (row["Begin Time (s)"] - annot["Begin Time (s)"])
            bool2 = (
                row["Begin Time (s)"] - annot["Begin Time (s)"]
            ) < self.n_prec_preds * conf.CONTEXT_WIN / conf.SR
            self.prec_anns = annot.loc[bool1 * bool2]
            if len(self.prec_anns) > self.n_exceed_thresh:
                self.filtered_annots = pd.concat(
                    [self.filtered_annots, self.prec_anns]
                )
                # this stops the function as soon as the limit is met once
                if not self.return_counts:
                    return 1
        if len(self.filtered_annots) > 0:
            self.filtered_annots = (self.filtered_annots
                                    .drop_duplicates()
                                    .sort_values(['Begin Time (s)']))
        return len(self.filtered_annots)

    def get_end_of_last_annotation(self):
        """
        Get the time corresponding to the last annotation in current file.
        """
        if len(self.annot_all) == 0:
            self.end = False
        else:
            self.end = int(self.annot_all["End Time (s)"].iloc[-1])

    def filter_files_of_hour_by_limit(self, date, hour):
        """
        Process the annotation counts and binary presence for a given
        hour in the dataset. This function is quite cryptic because there
        are several dataframes that have to be updated to insert the
        correct number of annotations and the binary presence value
        for the given hour and date.

        Parameters
        ----------
        date : string
            date string
        hour : string
            hour string
        """
        for h in range(0, self.end or 1, 3600):
            fil_h_ann = self.annot_all.loc[
                (h < self.annot_all["Begin Time (s)"])
                & (self.annot_all["Begin Time (s)"] < h + 3600)
            ]
            date, hour = init_new_dt_if_exceeding_3600_s(h, date, hour)

            fil_h_ann = fil_h_ann.loc[
                fil_h_ann[conf.ANNOTATION_COLUMN] >= self.thresh
            ]
            if not date in self.df["Date"].values:
                if not self.row == 0:
                    self.df.loc[self.row, conf.HR_DP_COL] = daily_prs(self.df)
                    self.df_counts.loc[self.row, conf.HR_DA_COL] = sum(
                        self.df_counts.loc[
                            len(self.df_counts), h_of_day_str()
                        ].values
                    )

                    if self.sc:
                        self.df_sc.loc[self.row, conf.HR_DP_COL] = daily_prs(
                            self.df_sc
                        )
                        self.df_sc_cnt.loc[self.row, conf.HR_DA_COL] = sum(
                            self.df_sc_cnt.loc[
                                len(self.df_sc_cnt), h_of_day_str()
                            ].values
                        )

                self.row += 1
                self.df.loc[self.row, "Date"] = date
                self.df_counts.loc[self.row, "Date"] = date
                if self.sc:
                    self.df_sc.loc[self.row, "Date"] = date
                    self.df_sc_cnt.loc[self.row, "Date"] = date

            self.df.loc[self.row, hour] = hourly_prs(fil_h_ann, lim=self.lim)
            self.df_counts.loc[self.row, hour] = len(fil_h_ann)

            if self.file_ind == len(self.files):
                self.df.loc[self.row, conf.HR_DP_COL] = daily_prs(self.df)
                self.df_counts.loc[self.row, conf.HR_DA_COL] = sum(
                    self.df_counts.loc[
                        len(self.df_counts), h_of_day_str()
                    ].values
                )

                if self.sc:
                    self.df_sc.loc[self.row, conf.HR_DP_COL] = daily_prs(
                        self.df_sc
                    )
                    self.df_sc_cnt.loc[self.row, conf.HR_DA_COL] = sum(
                        self.df_sc_cnt.loc[
                            len(self.df_sc_cnt), h_of_day_str()
                        ].values
                    )

            if self.sc:
                self.df_sc_cnt.loc[self.row, hour] = self.seq_crit(fil_h_ann)
                self.df_sc.loc[self.row, hour] = int(
                    bool(self.df_sc_cnt.loc[self.row, hour])
                )

    def save_filtered_selection_tables(self, dataset_path):
        """
        Save the selection tables under a new directory with the
        chosen filter settings. Depending if sequence limit is chosen
        or not a directory name is chosen and saved in the parent
        timestamp foldername of the current inference session.

        Parameters
        ----------
        dataset_path : pathlib.Path
            path to dataset in current annotation timestamp folder
        """
        if self.sc:
            thresh_label = f"thresh_{self.thresh_sc}_seq_{self.lim_sc}"
        else:
            thresh_label = f"thresh_{self.thresh}_sim"
        conf.THRESH_LABEL = thresh_label
        new_thresh_path = Path(conf.GEN_ANNOT_SRC).joinpath(thresh_label)
        new_thresh_path = new_thresh_path.joinpath(
            self.files[self.file_ind - 1]
            .relative_to(dataset_path.parent)
            .parent
        )
        new_thresh_path.mkdir(exist_ok=True, parents=True)
        file_path = new_thresh_path.joinpath(
            self.files[self.file_ind - 1].stem
            + self.files[self.file_ind - 1].suffix
        )
        if not self.sc:
            self.filtered_annots = self.annot_all.loc[
                self.annot_all[conf.ANNOTATION_COLUMN] >= self.thresh
            ]
        if len(self.filtered_annots) > 0:
            self.filtered_annots.index = self.filtered_annots.Selection
            self.filtered_annots.pop("Selection")
            self.filtered_annots.to_csv(file_path, sep="\t")

    def update_annotation_progbar(self, **kwargs):
        """
        Update the annotation progbar in the corresponding streamlit widget.
        """
        import streamlit as st

        inner_counter = self.file_ind / len(self.files)
        outer_couter = self.dir_ind / self.total_dirs
        counter = inner_counter * 1 / self.total_dirs + outer_couter

        if "preset" in kwargs:
            st.session_state.progbar_update.progress(
                counter,
                text="Progress",
            )
            if counter == 1 and "update_plot" in kwargs:
                st.write("Plot updated")
                st.button("Update plot")
        elif conf.PRESET == 3:
            kwargs["progbar1"].progress(
                counter,
                text="Progress",
            )


def return_hourly_pres_df(
    files,
    thresh,
    thresh_sc,
    lim,
    lim_sc,
    sc,
    path,
    total_dirs,
    dir_ind,
    return_counts=True,
    save_filtered_selection_tables=False,
    **kwargs,
):
    """
    Return the hourly presence and hourly annotation counts for all files
    within the chosen dataset. Processing is handled by the ProcessLimits class
    this is the caller function.

    Parameters
    ----------
    files : list
        pathlib.Path objects linking to files
    thresh : float
        threshold
    thresh_sc : float
        threshold for sequence limit
    lim : int
        limit of simple limit for binary presence
    lim_sc : int
        limit for sequence limit
    sc : bool
        sequence limit yes or no
    path : pathlib.Path
        path to current dataset in annotations folder
    total_dirs : int
        number of directories to be annotated
    dir_ind : int
        index of directory
    return_counts : bool, optional
        only binary presence or hourly counts, by default True
    save_filtered_selection_tables : bool, optional
        whether to save the filtered selection tables or not, by default False

    Returns
    -------
    ProcessLimits object
        contains all dataframes with the hourly and daily metrics
    """
    if not isinstance(path, Path):
        path = Path(path)

    tup, counts = init_date_tuple(files)
    filt_annots = ProcessLimits(
        files,
        thresh,
        thresh_sc,
        lim,
        lim_sc,
        sc,
        dir_ind,
        total_dirs,
        return_counts,
    )
    for (date, hour), count in zip(tup, counts):
        filt_annots.concat_files_within_hour(count)

        filt_annots.get_end_of_last_annotation()

        filt_annots.filter_files_of_hour_by_limit(date, hour)

        if save_filtered_selection_tables:
            filt_annots.save_filtered_selection_tables(path)

        print(
            f"Computing files in {path.stem}: "
            f"{filt_annots.file_ind}/{len(files)}",
            end="\r",
        )
        if "preset" in kwargs or conf.PRESET == 3 and conf.STREAMLIT:
            filt_annots.update_annotation_progbar(**kwargs)

    return filt_annots


def get_path(path, metric):
    if not path.stem == "analysis":
        save_path = path.parent.joinpath("analysis").joinpath(path.stem)
    else:
        save_path = path
    save_path.mkdir(exist_ok=True, parents=True)
    return save_path.joinpath(f"{metric}.csv")


def get_title(metric):
    if "annotation" in metric:
        return "Annotation counts for each hour"
    elif "presence" in metric:
        return "Hourly presence"


def plot_hp(path, lim, thresh, metric):
    path = get_path(path, metric)
    df = pd.read_csv(path)
    h_pres = df.loc[:, h_of_day_str()]
    h_pres.index = df["Date"]
    plt.figure(figsize=[8, 6])
    plt.title(
        f"{get_title(metric)}, limit={lim:.0f}, " f"threshold={thresh:.2f}"
    )
    if "presence" in metric:
        d = {"vmin": 0, "vmax": 1}
    else:
        d = {"vmax": conf.HR_CNTS_VMAX}
    sns.heatmap(h_pres.T, cmap="crest", **d)
    plt.ylabel("hour of day")
    plt.tight_layout()
    plt.savefig(
        path.parent.joinpath(f"{metric}_{thresh:.2f}_{lim:.0f}.png"), dpi=150
    )
    plt.close()


def calc_val_diff(
    time_dir=None,
    thresh=conf.THRESH,
    lim=conf.SIMPLE_LIMIT,
    thresh_sc=conf.SEQUENCE_THRESH,
    lim_sc=conf.SEQUENCE_LIMIT,
    sc=True,
    **kwargs,
):
    path = find_thresh05_path_in_dir(time_dir)
    for ind, fold in enumerate(path.iterdir()):
        if not fold.joinpath("analysis").joinpath(conf.HR_VAL_PATH).exists():
            continue

        df_val = get_val(fold.joinpath("analysis").joinpath(conf.HR_VAL_PATH))
        hours_of_day = ["%.2i:00" % i for i in np.arange(24)]
        files = get_files(
            location=path, search_str="**/*txt"
        )
        files.sort()

        annots = return_hourly_pres_df(
            files,
            thresh,
            thresh_sc,
            lim,
            lim_sc,
            sc,
            fold,
            total_dirs=len(list(path.iterdir())),
            dir_ind=ind + 1,
            return_counts=False,
            **kwargs,
        )

        d, incorrect, df_diff = dict(), dict(), dict()
        for agg_met, df_metric in zip(("sl", "sq"), (annots.df, annots.df_sc)):
            df_val.index = df_metric.index
            df_diff.update(
                {
                    # the calculation will result in tp being 2, fp -1, fn 1 and tn 0
                    agg_met:
                        (
                            df_val.loc[:, hours_of_day] - df_metric.loc[:, hours_of_day] 
                            + 2 * df_val.loc[:, hours_of_day] * df_metric.loc[:, hours_of_day]
                        )
                }
            )

            results = np.unique(df_diff[agg_met])
            d.update(
                {agg_met: dict({"true_pos": 0, "false_pos": 0, "false_neg": 0, "true_neg": 0})}
            )
            
            if 2 in results:
                d[agg_met]['true_pos'] = len(np.where(df_diff[agg_met] == 2)[0])
            if -1 in results:
                d[agg_met]['false_pos'] = len(np.where(df_diff[agg_met]== -1)[0])
            if 1 in results:
                d[agg_met]['false_neg'] = len(np.where(df_diff[agg_met] == 1)[0])
            if 0 in results:
                d[agg_met]['true_neg'] = len(np.where(df_diff[agg_met] == 0)[0])
            
            incorrect.update(
                {agg_met: d[agg_met]["false_pos"] + d[agg_met]["false_neg"]}
            )
            d[agg_met]['precision'] = d[agg_met]['true_pos'] / (d[agg_met]['false_pos'] + d[agg_met]['true_pos'])
            d[agg_met]['recall'] = d[agg_met]['true_pos'] / (d[agg_met]['false_neg'] + d[agg_met]['true_pos'])
        perf_df = pd.DataFrame(d)

        print(
            "\n",
            "l:",
            lim,
            "th:",
            thresh,
            "incorrect:",
            incorrect["sl"],
            # "%.2f" % (incorrect["sl"] / (len(df_diff["sl"]) * 24) * 100),
        )
        print(
            "l:",
            lim_sc,
            "th:",
            thresh_sc,
            "sc_incorrect:",
            incorrect["sq"],
            # "%.2f" % (incorrect["sq"] / (len(df_diff["sl"]) * 24) * 100),
        )

        save_dir = fold.joinpath(files[-1].parent.stem).joinpath('validation')
        save_dir.mkdir(exist_ok=True, parents=True)
        
        annots.df.to_csv(
            save_dir.joinpath(f"th{thresh}_l{lim}_hourly_pres_sl.csv")
        )
        annots.df_sc.to_csv(
            save_dir.joinpath(f"th{thresh_sc}_l{lim_sc}_hourly_pres_sequ_crit.csv")
        )
        df_diff["sl"].to_csv(
            save_dir.joinpath(f"th{thresh}_l{lim}_diff_hourly_pres_sl.csv")
        )
        df_diff["sq"].to_csv(
            save_dir.joinpath(
                f"th{thresh_sc}_l{lim_sc}_diff_hourly_pres_sequ_crit.csv"
            )
        )
        perf_df.to_csv(
            save_dir.joinpath(f"th{thresh_sc}_l{lim_sc}_diff_performance.csv")
        )


def plot_varying_limits(annotations_path=conf.ANNOT_DEST):
    thresh_sl, thresh_sc = 0.9, 0.9
    for lim_sl, lim_sc in zip(np.linspace(10, 48, 20), np.linspace(1, 20, 20)):
        for lim, thresh in zip((lim_sl, thresh_sl), (lim_sc, thresh_sc)):
            compute_hourly_pres(
                annotations_path,
                thresh_sc=thresh_sc,
                lim_sc=lim_sc,
                thresh=thresh,
                lim=lim,
            )
            for metric in (
                conf.HR_CNTS_SC,
                conf.HR_CNTS_SL,
                conf.HR_PRS_SC,
                conf.HR_PRS_SL,
            ):
                plot_hp(annotations_path, lim, thresh, metric)
