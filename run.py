def main(sc=True, **kwargs):
    """
    Main function to run the whole pipeline. The function is called from the
    streamlit app. The function is called with the preset option as an argument.
    The preset option is used to determine which function should be called.
    The preset option is set in the config file.

    Parameters
    ----------
    sc : bool, optional
        Decide if meta analysis should be done using sequence limit, by default True

    Returns
    -------
    str or None
        depending on the preset option, the function returns either the time
        when the annotation was started or None
    """
    from acodet.annotate import run_annotation, filter_annots_by_thresh
    from acodet.train import run_training, save_model
    from acodet.tfrec import write_tfrec_dataset
    from acodet.hourly_presence import compute_hourly_pres, calc_val_diff
    from acodet.evaluate import create_overview_plot
    from acodet.combine_annotations import generate_final_annotations
    from acodet.models import init_model
    import acodet.global_config as conf

    if "fetch_config_again" in kwargs:
        import importlib

        importlib.reload(conf)
        kwargs["relativ_path"] = conf.SOUND_FILES_SOURCE
    if "preset" in kwargs:
        preset = kwargs["preset"]
    else:
        preset = conf.PRESET

    if conf.RUN_CONFIG == 1:
        if preset == 1:
            timestamp_foldername = run_annotation(**kwargs)
            return timestamp_foldername
        elif preset == 2:
            new_thresh = filter_annots_by_thresh(**kwargs)
            return new_thresh
        elif preset == 3:
            compute_hourly_pres(sc=sc, **kwargs)
        elif preset == 4:
            compute_hourly_pres(**kwargs)
        elif preset == 6:
            calc_val_diff(**kwargs)
        elif preset == 0:
            timestamp_foldername = run_annotation(**kwargs)
            filter_annots_by_thresh(timestamp_foldername, **kwargs)
            compute_hourly_pres(timestamp_foldername, sc=sc, **kwargs)
            return timestamp_foldername

    elif conf.RUN_CONFIG == 2:
        if preset == 1:
            generate_final_annotations(**kwargs)
            write_tfrec_dataset(**kwargs)
        elif preset == 2:
            generate_final_annotations(active_learning=False, **kwargs)
            write_tfrec_dataset(active_learning=False, **kwargs)

    elif conf.RUN_CONFIG == 3:
        if preset == 1:
            run_training(**kwargs)
        elif preset == 2:
            create_overview_plot(**kwargs)
        elif preset == 3:
            create_overview_plot("2022-05-00_00", **kwargs)
        elif preset == 4:
            save_model("FlatHBNA", init_model(), **kwargs)


if __name__ == "__main__":
    from acodet.create_session_file import create_session_file

    create_session_file()
    main()
