def main(**kwargs):
    from acodet.annotate import run_annotation, filter_annots_by_thresh
    from acodet.train import run_training, save_model
    from acodet.tfrec import write_tfrec_dataset
    from acodet.hourly_presence import compute_hourly_pres, calc_val_diff
    from acodet.evaluate import create_overview_plot
    from acodet.combine_annotations import generate_final_annotations
    from acodet.models import init_model
    import acodet.global_config as conf
    
    if conf.RUN_CONFIG == 1:
        if conf.PRESET == 1:
            run_annotation(**kwargs)
        elif conf.PRESET == 2:
            filter_annots_by_thresh(**kwargs)
        elif conf.PRESET == 3:
            compute_hourly_pres(sc=True, **kwargs)
        elif conf.PRESET == 4:
            compute_hourly_pres(**kwargs)
        elif conf.PRESET == 5:
            pass # TODO hourly preds mit varying limits
        elif conf.PRESET == 6:
            calc_val_diff(**kwargs)
        elif conf.PRESET == 0:
            time_start = run_annotation(**kwargs)
            filter_annots_by_thresh(time_start, **kwargs)
            compute_hourly_pres(time_start, sc=True, **kwargs)
        
    elif conf.RUN_CONFIG == 2:
        if conf.PRESET == 1:
            generate_final_annotations(**kwargs)
            write_tfrec_dataset(**kwargs)
        elif conf.PRESET == 2:
            generate_final_annotations(active_learning=False, **kwargs)
            write_tfrec_dataset(active_learning=False, **kwargs)
            
    elif conf.RUN_CONFIG == 3:
        if conf.PRESET == 1:
            run_training(**kwargs)
        elif conf.PRESET == 2:
            create_overview_plot(**kwargs)
        elif conf.PRESET == 3:
            create_overview_plot('2022-05-00_00', **kwargs)
        elif conf.PRESET == 4:
            save_model('FlatHBNA', init_model(), **kwargs)
            
if __name__ == '__main__':
    from acodet.create_session_file import create_session_file
    create_session_file()
    main()