from hbdet.annotate import run_annotation, filter_annots_by_thresh
from hbdet.train import run_training, save_model
from hbdet.tfrec import write_tfrec_dataset
from hbdet.hourly_presence import compute_hourly_pres
from hbdet.evaluate import create_overview_plot
from hbdet.combine_annotations import generate_final_annotations
from hbdet.models import init_model
import hbdet.global_config as conf

if conf.RUN_CONFIG == 1:
    if conf.PRESET == 1:
        run_annotation()
    elif conf.PRESET == 2:
        filter_annots_by_thresh()
    elif conf.PRESET == 3:
        pass # TODO mit sequ crit filtern
    elif conf.PRESET == 4:
        compute_hourly_pres()
    elif conf.PRESET == 5:
        pass # TODO hourly preds mit varying limits
    elif conf.PRESET == 0:
        time_start = run_annotation()
        filter_annots_by_thresh(time_start)
        compute_hourly_pres(time_start)
    
elif conf.RUN_CONFIG == 2:
    if conf.PRESET == 1:
        generate_final_annotations()
        write_tfrec_dataset()
    elif conf.PRESET == 2:
        generate_final_annotations()
        generate_final_annotations(active_learning=False)
        
elif conf.RUN_CONFIG == 3:
    if conf.PRESET == 1:
        run_training()
    elif conf.PRESET == 2:
        create_overview_plot()
    elif conf.PRESET == 3:
        create_overview_plot('2022-05-00_00')
    elif conf.PRESET == 4:
        save_model('FlatHBNA', init_model())