import streamlit as st
import run
import hbdet.global_config as conf
from pathlib import Path
from hbdet.front_end import utils

def train_options():
    preset_option = st.selectbox(   
        'How would you like run the program?',
        ('1 - train new model',
         '2 - continue training on existing model and save model in the end',
         '3 - evaluate saved model', 
         '4 - evaluate model checkpoint',
         '5 - save model specified in advanced config'))
    utils.next_button(id=3)
    if not st.session_state.b3:
        pass
    preset_option = int(preset_option[0])
    
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
        if not model_architecture == 'HumpBackNorthAtlantic':
            conf.KERAS_MOD_NAME = True
        st.markdown('#### Hyperparameters')
        utils.user_input('batch size', '32')
        utils.user_input('epochs', '50')
        utils.user_input('steps per epoch', '1000')
        utils.user_input('initial learning rate', '0.0005')
        utils.user_input('final learning rate', '0.000005')
        st.markdown('#### Augmentations')
        utils.user_dropdown('Use time-shift augmentation', ('True', 'False'))
        utils.user_dropdown('Use mixup augmentation', ('True', 'False'))
        utils.user_dropdown('Use specaugment augmentation', ('True', 'False'))

    
    return preset_option

