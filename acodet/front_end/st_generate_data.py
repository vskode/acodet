import streamlit as st
import run
import acodet.global_config as conf
from pathlib import Path
from acodet.front_end import utils
import json
help_str = json.load(open("acodet/front_end/help_strings.json", 'rb'))

def generate_data_options(key='gen_data'):
    preset_option = st.selectbox(
        'How would you like run the program?',
        ('1 - generate new training data from reviewed annotations',
         '2 - generate new training data from reviewed annotations '
         'and fill space between annotations with noise annotations'),
        key = 'gen_data')
    utils.next_button(id=2)
    if not st.session_state.b2:
        pass
    preset_option = int(preset_option[0])
    
    if preset_option == 1:
        utils.open_folder_dialogue(st.text_input(
            "Enter the path to your sound data:", '.'), 
                            key='source_folder_' + key)
        utils.open_folder_dialogue(st.text_input(
            "Enter the path to your reviewed annotations:", '.'), 
                            key='reviewed_annotations_folder' + key)
        
        st.markdown("### Settings")
        st.markdown("#### Audio preprocessing")
        utils.user_input("sample rate [Hz]", "2000")
        utils.user_input("context window length [s]", "3.9")
        st.markdown("#### Spectrogram settings")
        utils.user_input("STFT frame length [samples]", "1024")
        utils.user_input("number of frequency bins", "128")
        st.markdown('#### TFRecord creationg settings')
        utils.user_input("limit of context windows per tfrecord file", "600")
        utils.user_input("trainratio", "0.7")
        utils.user_input("test validation ratio", "0.7")
        
    elif preset_option == 2:
        utils.open_folder_dialogue(st.text_input(
            "Enter the path to your sound data:", '.'), 
                            key='source_folder_' + key)
        utils.open_folder_dialogue(st.text_input(
            "Enter the path to your reviewed annotations:", '.'), 
                            key='reviewed_annotations_folder' + key)

        st.markdown("### Settings")
        st.markdown("#### Audio preprocessing")
        utils.user_input("sample rate [Hz]", "2000")
        utils.user_input("context window length [s]", "3.9")
        st.markdown("#### Spectrogram settings")
        utils.user_input("STFT frame length [samples]", "1024")
        utils.user_input("number of frequency bins", "128")
        st.markdown('#### TFRecord creationg settings')
        utils.user_input("limit of context windows per tfrecord file", "600")
        utils.user_input("trainratio", "0.7")
        utils.user_input("test validation ratio", "0.7")

    return preset_option