import streamlit as st
from acodet.create_session_file import create_session_file
if not 'session_started' in st.session_state:
    st.session_state.session_started = True
    create_session_file()
from acodet.front_end import (utils, st_annotate, st_generate_data, st_train)

def select_preset(run_option):
    run_option = int(run_option[0])
    utils.write_to_session_file('run_config', run_option)
    show_run_btn  = False
    
    if run_option == 1:
        show_run_btn = st_annotate.annotate_options()
        kwargs = {'callbacks': [utils.CustomCallback()]}
    elif run_option == 2:
        show_run_btn = st_generate_data.generate_data_options()
        kwargs = dict()
    if run_option == 3:
        show_run_btn = st_train.train_options()
        kwargs = dict()
    if show_run_btn:
        run_computions(**kwargs)

def run_computions(**kwargs):
    if not st.button('Run'):
        pass
    else:
        import run
        st.write('Program started')
        run.main(**kwargs)
        print('finished')
        st.stop()

run_option = st.selectbox(
    'How would you like run the program?',
    ('1 - Annotate', 
     '2 - Generate new training data', 
     '3 - Train'), 
    key = 'main',
    help="you're being helped")
select_preset(run_option)

    
