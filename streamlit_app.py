import streamlit as st
from acodet.create_session_file import create_session_file, read_session_file

if not "session_started" in st.session_state:
    st.session_state.session_started = True
    create_session_file()
from acodet.front_end import (
    utils,
    st_annotate,
    st_generate_data,
    st_train,
    st_visualization,
)

utils.write_to_session_file("streamlit", True)


def select_preset():
    utils.write_to_session_file("run_config", st.session_state.run_option)
    show_run_btn = False

    show_run_btn = st_annotate.annotate_options()
    
    if show_run_btn:
        run_computions()


def run_computions(**kwargs):
    utils.next_button(id=4, text="Run computations")
    if st.session_state.b4:
        display_not_implemented_text()
        kwargs = utils.prepare_run()
        if not st.session_state.run_finished:
            import run

            st.session_state.save_dir = run.main(
                fetch_config_again=True, update_plot=False, **kwargs
            )
            st.session_state.run_finished = True

    if st.session_state.run_finished:
        if not st.session_state.preset_option == 3:
            st.write("Computation finished")
            utils.next_button(id=5, text="Show results")
            st.markdown("""---""")
        else:
            conf = read_session_file()
            st.session_state.b5 = True
            st.session_state.save_dir = conf["generated_annotation_source"]

        if not st.session_state.b5:
            pass
        else:
            st_visualization.output()
            st.stop()


def display_not_implemented_text():
    if not st.session_state.run_option == 1:
        st.write(
            """This option is not yet implemented for usage
                    with the user interface. A headless version is
                    available at https://github.com/vskode/acodet."""
        )
        st.stop()


if __name__ == "__main__":
    
    st.markdown(
        """
        # Welcome to AcoDet - Acoustic Detection of Animal Vocalizations :loud_sound:
        ### This program can be used to generate predictions from bioacoustic deep 
        learning models. AcoDet supports models implemented in bacpipe that have 
        available classification heads. In the future, custom classification heads
        will be able to add to the feature extractors. 
        
        For more information, please visit https://github.com/vskode/acodet
        
        ---
        """
    )
    run_option = 1

    st.session_state.run_option = run_option
    select_preset()
