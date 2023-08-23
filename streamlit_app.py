import streamlit as st
from acodet.create_session_file import create_session_file
if not 'session_started' in st.session_state:
    st.session_state.session_started = True
    create_session_file()
from acodet.front_end import (utils, st_annotate, st_generate_data, st_train, visualization)

utils.write_to_session_file('streamlit', True)

def select_preset():
    utils.write_to_session_file('run_config', st.session_state.run_option)
    show_run_btn  = False
    
    if st.session_state.run_option == 1:
        show_run_btn = st_annotate.annotate_options()
    elif st.session_state.run_option == 2:
        show_run_btn = st_generate_data.generate_data_options()
    if st.session_state.run_option == 3:
        show_run_btn = st_train.train_options()
    if show_run_btn:
        run_computions()

def run_computions(**kwargs):
    utils.next_button(id=4, text='Run computations')
    if not st.session_state.b4:
        st.session_state.run_finished = False
    else:
        kwargs = utils.prepare_run()
        if not st.session_state.run_finished:
            import run
            st.session_state.time_dir = run.main(**kwargs)
            st.session_state.run_finished = True
        
    if st.session_state.run_finished:
        st.write('Computation finished')
        utils.next_button(id=5, text='Show results')
        if not st.session_state.b5:
            pass
        else:
            visualization.output()
            st.stop()

run_option = st.selectbox(
    'How would you like run the program?',
    ('1 - Annotate', 
     '2 - Generate new training data', 
     '3 - Train'), 
    key = 'main',
    help="you're being helped")

st.session_state.run_option = int(run_option[0])
select_preset()

    
