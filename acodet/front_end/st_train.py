import streamlit as st
from acodet.front_end import utils

def train_options(key='train'):
    preset_option = int(st.selectbox(   
        'How would you like run the program?',
        ('1 - train new model',
         '2 - continue training on existing model and save model in the end',
         '3 - evaluate saved model', 
         '4 - evaluate model checkpoint',
         '5 - save model specified in advanced config'),
        key=key)[0])
    st.session_state.preset_option = preset_option
    utils.make_nested_btn_false_if_dropdown_changed(3, preset_option, 3)
    utils.next_button(id=3)
    if not st.session_state.b3:
        pass
    else:
        config = dict()
        config['predefined_settings'] = preset_option
        
        if preset_option == 1:
            st.markdown('### Model training settings')
            st.markdown('#### Model architecture')
            model_architecture = utils.user_dropdown(
                'Which model architecture would you like to use?',
                ('HumpBackNorthAtlantic',
                'ResNet50', 'ResNet101',
                'ResNet152', 'MobileNet',
                'DenseNet169', 'DenseNet201',
                'EfficientNet0', 'EfficientNet1', 
                'EfficientNet2', 'EfficientNet3', 
                'EfficientNet4', 'EfficientNet5', 
                'EfficientNet6', 'EfficientNet7')
            )
            config['ModelClassName'] = model_architecture
            if not model_architecture == 'HumpBackNorthAtlantic':
                config['keras_mod_name'] = True
            st.markdown('#### Hyperparameters')
            config['batch_size'] = utils.validate_int(utils.user_input('batch size', '32'))
            config['epochs'] = utils.validate_int(utils.user_input('epochs', '50'))
            config['steps_per_epoch'] = utils.validate_int(utils.user_input('steps per epoch', '1000'))
            config['init_lr'] = utils.validate_float(utils.user_input('initial learning rate', '0.0005'))
            config['final_lr'] = utils.validate_float(utils.user_input('final learning rate', '0.000005'))
            st.markdown('#### Augmentations')
            config['time_augs'] = utils.user_dropdown('Use time-shift augmentation', ('True', 'False'))
            config['mixup_augs'] = utils.user_dropdown('Use mixup augmentation', ('True', 'False'))
            config['spec_aug'] = utils.user_dropdown('Use specaugment augmentation', ('True', 'False'))
        
            for key in ['time_augs', 'mixup_augs', 'spec_aug']:
                if config[key] == 'False':
                    config[key] = False
                else:
                    config[key] = True
        
        for k, v in config.items():
            utils.write_to_session_file(k, v)
        return True
