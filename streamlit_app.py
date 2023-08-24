import streamlit as st
from acodet.create_session_file import create_session_file, read_session_file
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
    elif st.session_state.run_option == 3:
        show_run_btn = st_train.train_options()
    if show_run_btn:
        run_computions()

def run_computions(**kwargs):
    if not st.session_state.preset_option == 3:
        utils.next_button(id=4, text='Run computations')
        if not st.session_state.b4:
            st.session_state.run_finished = False
        else:
            kwargs = utils.prepare_run()
            if not st.session_state.run_finished:
                import run
                st.session_state.save_dir = run.main(**kwargs)
                st.session_state.run_finished = True
    else:
        st.session_state.run_finished = True
        
    if st.session_state.run_finished:
        if not st.session_state.preset_option == 3:
            st.write('Computation finished')
            utils.next_button(id=5, text='Show results')
            st.markdown("""---""")
        else:
            conf = read_session_file()
            st.session_state.b5 = True
            st.session_state.save_dir = conf['generated_annotation_source']

        if not st.session_state.b5:
            pass
        else:
            visualization.output()
            st.stop()

if __name__ == '__main__':
    run_option = int(st.selectbox(
        'How would you like run the program?',
        ('1 - Annotate', 
        '2 - Generate new training data', 
        '3 - Train'), 
        key = 'main',
        help="you're being helped")[0])
    
    st.session_state.run_option = run_option
    select_preset()

    
