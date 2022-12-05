from train import run_training
from hbdet.models import init_model
import hbdet.global_config as conf
from annotate import run_annotation, generate_stats
from combine_annotations import generate_final_annotations


if conf.RUN_CONFIG == 1:
    run_annotation()
    
elif conf.RUN_CONFIG == 2:
    generate_stats()
    
elif conf.RUN_CONFIG == 3:
    generate_final_annotations()
    
elif conf.RUN_CONFIG == 4:
    run_training()