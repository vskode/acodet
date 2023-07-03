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
    utils.next_button(id=1)
    if not st.session_state.b1:
        pass
    preset_option = int(preset_option[0])
    
    if preset_option == 1 or preset_option == 0:
        path = st.text_input("Enter the path to your sound data:", '.')
        utils.open_folder_dialogue(path, key='folder_' + key)
        utils.user_input("Model threshold:", "0.9")
        st.markdown("## Aggregation metrics parameters for hourly presence and hourly counts.")
        st.markdown('### Specify parameters for the simple limit.')
        utils.user_input("Number of annotations for simple limit:", "15")
        utils.user_input("Threshold for simple limit:", "0.9")
        st.markdown('### Specify parameters for the sequence limit.')
        utils.user_input("Number of annotations for sequence limit:", "20")
        utils.user_input("Threshold for sequence limit:", "0.9")

    elif preset_option == 2:
        path = st.text_input("Enter the path to your annotation data:", '.')
        utils.open_folder_dialogue(path, key='folder_' + key)
        utils.user_input("Rerun annotations Model threshold:", "0.9")
        
    else:
        st.markdown("## Aggregation metrics parameters for hourly presence and hourly counts.")
        st.markdown('### Specify parameters for the simple limit.')
        utils.user_input("Number of annotations for simple limit:", "15")
        utils.user_input("Threshold for simple limit:", "0.9")
        st.markdown('### Specify parameters for the sequence limit.')
        utils.user_input("Number of annotations for sequence limit:", "20")
        utils.user_input("Threshold for sequence limit:", "0.9")
    
    return preset_option
