import train
from hbdet.models import GoogleMod
from hbdet.funcs import init_model

model_class, load_ckpt_path, keras_mod_name, load_g_ckpt = GoogleMod, '2022-11-30_01', False, False
model = init_model(model_class, 
            f'../trainings/{load_ckpt_path}/unfreeze_no-TF', 
            keras_mod_name=keras_mod_name, input_specs=True,
            load_g_ckpt=load_g_ckpt)
train.save_model('GoogleModel', model)