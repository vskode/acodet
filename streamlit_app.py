import streamlit as st
import run
import hbdet.global_config as conf
from pathlib import Path
from hbdet.front_end import utils


def annotate_options(key='annot'):
    preset_option = st.selectbox(
        'What predefined Settings would you like to run?',
        ('1 - generate new annotations',
         '2 - filter existing annotations with different threshold',
         '3 - generate hourly predictions (simple limit and sequence criterion)',
         '4 - generate hourly predictions (only simple limit)',
         '0 - all of the above'), key=key)
    if not st.button('Next', key = 'button_' + key):
        pass
    preset_option = int(preset_option[0])
    
    if preset_option == 1 or preset_option == 0:
        path = st.text_input("Enter the path to your sound data:", '.')
        utils.open_folder_dialogue(path, key='folder_' + key)
        
        if not st.button('Next', key = 'button_' + key):
            pass
        st.text_input("Model threshold:", "0.9")
        st.text_input("Model threshold:", "0.9")
    
    
    
    return preset_option

def generate_data_options(key='gen_data'):
    preset_option = st.selectbox(
        'How would you like run the program?',
        ('1 - generate new training data from reviewed annotations',
         '2 - generate new training data from reviewed annotations '
         'and fill space between annotations with noise annotations'),
        key = 'gen_data')
    if not st.button('Next', key = 'button_' + key):
        pass

    st.write('You selected:', preset_option)
    preset_option = int(preset_option[0])
    
    if preset_option == 1:
        utils.open_folder_dialogue(st.text_input(
            "Enter the path to your sound data:", '.'), 
                            key='source_folder_' + key)
        utils.open_folder_dialogue(st.text_input(
            "Enter the path to your reviewed annotations:", '.'), 
                            key='reviewed_annotations_folder' + key)
    elif preset_option == 2:
        utils.open_folder_dialogue(st.text_input(
            "Enter the path to your sound data:", '.'), 
                            key='source_folder_' + key)
        utils.open_folder_dialogue(st.text_input(
            "Enter the path to your reviewed annotations:", '.'), 
                            key='reviewed_annotations_folder' + key)
    return preset_option

def train_options():
    preset_option = st.selectbox(   
        'How would you like run the program?',
        ('1 - continue training on existing model and save model in the end',
         '2 - evaluate saved model', 
         '3 - evaluate model checkpoint',
         '4 - save model specified in advanced config'))
    preset_option = int(preset_option[0])
    return preset_option


def select_preset(option):
    option = int(option[0])
    conf.RUN_CONFIG = option

    if option == 1:
        conf.PRESET = annotate_options()
    elif option == 2:
        conf.PRESET = generate_data_options()
    if option == 3:
        conf.PRESET = train_options()

def run_computions():
    if not st.button('Run'):
        pass
    st.write('Program started')
    run.main()
    st.stop()


option = st.selectbox(
    'How would you like run the program?',
    ('1 - Annotate', 
     '2 - Generate new training data', 
     '3 - Train'), 
    key = 'main',
    help="you're being helped")
if st.button('Next', key='button_main'):
    select_preset(option)

    
