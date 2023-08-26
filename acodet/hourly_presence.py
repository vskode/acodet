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
import time

time_start = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())


def hourly_prs(df, lim=10):
    if len(df) > lim:
        return 1
    else:
        return 0


def daily_prs(df):
    if 1 in df.loc[len(df), h_of_day_str()].values:
        return 1
    else:
        return 0


def get_val(path):
    df = pd.read_csv(path)
    return df


def seq_crit(
    annot,
    n_prec_preds=conf.SEQUENCE_CON_WIN,
    thresh_sc=0.9,
    n_exceed_thresh=4,
    return_counts=True,
):
    sequ_crit = 0
    annot = annot.loc[annot[conf.ANNOTATION_COLUMN] >= thresh_sc]
    for i, row in annot.iterrows():
        bool1 = 0 < (row["Begin Time (s)"] - annot["Begin Time (s)"])
        bool2 = (
            row["Begin Time (s)"] - annot["Begin Time (s)"]
        ) < n_prec_preds * conf.CONTEXT_WIN / conf.SR
        prec_anns = annot.loc[bool1 * bool2]
        if len(prec_anns) > n_exceed_thresh:
            sequ_crit += 1
            # this stops the function as soon as the limit is met once
            if not return_counts:
                return 1
    return sequ_crit


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
    if root.parts[-1] == "thresh_0.5":
        root = root.parent
    elif root.parts[-1] == "thresh_0.9":
        root = root.parent

    if not time_dir:
        path = root.joinpath("thresh_0.5")
    else:
        path = (
            Path(conf.GEN_ANNOTS_DIR).joinpath(time_dir).joinpath("thresh_0.5")
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
    return_hourly_counts=True,
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

    for ind, dir in enumerate(path.iterdir()):
        if not dir.is_dir():
            continue

        files = get_files(
            location=path.joinpath(dir.stem), search_str="**/*txt"
        )
        files.sort()

        df_tuple = return_hourly_pres_df(
            files,
            thresh,
            thresh_sc,
            lim,
            lim_sc,
            sc,
            dir,
            dir_ind=ind,
            total_dirs=len(list(path.iterdir())),
            **kwargs,
        )

        df, df_sc, df_counts, df_sc_counts = df_tuple

        df.to_csv(get_path(path.joinpath(dir.stem), conf.HR_PRS_SL))
        df_counts.to_csv(get_path(path.joinpath(dir.stem), conf.HR_CNTS_SL))
        if not "dont_save_plot" in kwargs.keys():
            for metric in (conf.HR_CNTS_SL, conf.HR_PRS_SL):
                plot_hp(path.joinpath(dir.stem), lim, thresh, metric)

        if sc:
            df_sc.to_csv(get_path(path.joinpath(dir.stem), conf.HR_PRS_SC))
            df_sc_counts.to_csv(
                get_path(path.joinpath(dir.stem), conf.HR_CNTS_SC)
            )
            if not "dont_save_plot" in kwargs.keys():
                for metric in (conf.HR_CNTS_SC, conf.HR_PRS_SC):
                    plot_hp(path.joinpath(dir.stem), lim_sc, thresh_sc, metric)
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
    **kwargs,
):
    if not isinstance(path, Path):
        path = Path(path)
    file_ind, row = 0, 0
    df = pd.DataFrame(columns=["Date", conf.HR_DP_COL, *h_of_day_str()])
    df_sc = df.copy()
    df_counts = pd.DataFrame(columns=["Date", conf.HR_DA_COL, *h_of_day_str()])
    df_sc_counts = df_counts.copy()
    tup, counts = init_date_tuple(files)
    for (date, hour), count in zip(tup, counts):
        annot_all = pd.DataFrame()

        for _ in range(count):
            annot_all = pd.concat(
                [annot_all, pd.read_csv(files[file_ind], sep="\t")]
            )
            file_ind += 1

        end = get_end_of_last_annotation(annot_all)

        for h in range(0, end or 1, 3600):
            annot = annot_all.loc[
                (h < annot_all["Begin Time (s)"])
                & (annot_all["Begin Time (s)"] < h + 3600)
            ]
            date, hour = init_new_dt_if_exceeding_3600_s(h, date, hour)

            annot = annot.loc[annot[conf.ANNOTATION_COLUMN] >= thresh]
            if not date in df["Date"].values:
                if not row == 0:
                    df.loc[row, conf.HR_DP_COL] = daily_prs(df)
                    df_counts.loc[row, conf.HR_DA_COL] = sum(
                        df_counts.loc[len(df_counts), h_of_day_str()].values
                    )

                    if sc:
                        df_sc.loc[row, conf.HR_DP_COL] = daily_prs(df_sc)
                        df_sc_counts.loc[row, conf.HR_DA_COL] = sum(
                            df_sc_counts.loc[
                                len(df_sc_counts), h_of_day_str()
                            ].values
                        )

                row += 1
                df.loc[row, "Date"] = date
                df_counts.loc[row, "Date"] = date
                if sc:
                    df_sc.loc[row, "Date"] = date
                    df_sc_counts.loc[row, "Date"] = date

            df.loc[row, hour] = hourly_prs(annot, lim=lim)
            df_counts.loc[row, hour] = len(annot)

            if file_ind == len(files):
                df.loc[row, conf.HR_DP_COL] = daily_prs(df)
                df_counts.loc[row, conf.HR_DA_COL] = sum(
                    df_counts.loc[len(df_counts), h_of_day_str()].values
                )

                if sc:
                    df_sc.loc[row, conf.HR_DP_COL] = daily_prs(df_sc)
                    df_sc_counts.loc[row, conf.HR_DA_COL] = sum(
                        df_sc_counts.loc[
                            len(df_sc_counts), h_of_day_str()
                        ].values
                    )

            if sc:
                df_sc_counts.loc[row, hour] = seq_crit(
                    annot,
                    thresh_sc=thresh_sc,
                    n_exceed_thresh=lim_sc,
                    return_counts=return_counts,
                )
                df_sc.loc[row, hour] = int(bool(df_sc_counts.loc[row, hour]))

        print(
            f"Computing files in {path.stem}: " f"{file_ind}/{len(files)}",
            end="\r",
        )
        if "preset" in kwargs or conf.PRESET == 3 and conf.STREAMLIT:
            import streamlit as st

            inner_counter = file_ind / len(files)
            outer_couter = dir_ind / total_dirs
            counter = inner_counter * 1 / total_dirs + outer_couter

            if "preset" in kwargs:
                st.session_state.progbar_update.progress(
                    counter,
                    text="Updating plot",
                )
                if counter == 1:
                    st.write("Plot updated")
            elif conf.PRESET == 3:
                kwargs["progbar1"].progress(
                    counter,
                    text="Updating plot",
                )
    return df, df_sc, df_counts, df_sc_counts


def get_path(path, metric):
    if not path.stem == "analysis":
        save_path = Path(path).joinpath("analysis")
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
    path = Path(path).joinpath("analysis")
    df = pd.read_csv(get_path(path, metric))
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
    plt.savefig(path.joinpath(f"{metric}_{thresh:.2f}_{lim:.0f}.png"), dpi=150)
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
            location=path.joinpath(fold.stem), search_str="**/*txt"
        )
        files.sort()

        df_tuple = return_hourly_pres_df(
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
        df, df_sc, _, _ = df_tuple

        d, incorrect, df_diff = dict(), dict(), dict()
        for agg_met, df_metric in zip(("sl", "sq"), (df, df_sc)):
            df_val.index = df_metric.index
            df_diff.update(
                {
                    agg_met: df_val.loc[:, hours_of_day]
                    - df_metric.loc[:, hours_of_day]
                }
            )

            results = np.unique(df_diff[agg_met])
            d.update(
                {agg_met: dict({"true": 0, "false_pos": 0, "false_neg": 0})}
            )
            for met, val in zip(d[agg_met].keys(), (0, -1, 1)):
                if val in results:
                    d[agg_met][met] = len(np.where(df_diff[agg_met] == val)[0])
            incorrect.update(
                {agg_met: d[agg_met]["false_pos"] + d[agg_met]["false_neg"]}
            )
        perf_df = pd.DataFrame(d)

        print(
            "\n",
            "l:",
            lim,
            "th:",
            thresh,
            "incorrect:",
            incorrect["sl"],
            "%.2f" % (incorrect["sl"] / (len(df_diff["sl"]) * 24) * 100),
        )
        print(
            "l:",
            lim_sc,
            "th:",
            thresh_sc,
            "sc_incorrect:",
            incorrect["sq"],
            "%.2f" % (incorrect["sq"] / (len(df_diff["sl"]) * 24) * 100),
        )

        df.to_csv(
            Path(fold)
            .joinpath("analysis")
            .joinpath(f"th{thresh}_l{lim}_hourly_presence.csv")
        )
        df_sc.to_csv(
            Path(fold)
            .joinpath("analysis")
            .joinpath(f"th{thresh_sc}_l{lim_sc}_hourly_pres_sequ_crit.csv")
        )
        df_diff["sl"].to_csv(
            Path(fold)
            .joinpath("analysis")
            .joinpath(f"th{thresh}_l{lim}_diff_hourly_presence.csv")
        )
        df_diff["sq"].to_csv(
            Path(fold)
            .joinpath("analysis")
            .joinpath(
                f"th{thresh_sc}_l{lim_sc}_diff_hourly_pres_sequ_crit.csv"
            )
        )
        perf_df.to_csv(
            Path(fold)
            .joinpath("analysis")
            .joinpath(f"th{thresh_sc}_l{lim_sc}_diff_performance.csv")
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
