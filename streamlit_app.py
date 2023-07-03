import streamlit as st
import run
import hbdet.global_config as conf
from pathlib import Path
from hbdet.front_end import (utils, st_annotate, st_generate_data, st_train)


def select_preset(option):
    option = int(option[0])
    conf.RUN_CONFIG = option

    if option == 1:
        conf.PRESET = st_annotate.annotate_options()
    elif option == 2:
        conf.PRESET = st_generate_data.generate_data_options()
    if option == 3:
        conf.PRESET = st_train.train_options()
    run_computions()

def run_computions():
    if not st.button('Run'):
        pass
    else:
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
utils.next_button(id=0)
if st.session_state.b0:
    select_preset(option)

    
