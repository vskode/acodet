import streamlit as st
from pathlib import Path
import pandas as pd
from acodet.create_session_file import read_session_file
conf = read_session_file()

def output():
    if st.session_state.run_option == 1:
        annots_path = Path(
            conf['generated_annotations_folder']
            ).joinpath(st.session_state.time_dir)
        st.markdown(
            f"""Your annotations are saved in the folder: 
            `{annots_path.resolve().as_posix()}`
            """)
        tab1, tab2 = st.tabs(['View overall statistics', 
                              'View individual annotation files'])
        with tab1:
            df = pd.read_csv(annots_path.joinpath('stats.csv'))
            st.dataframe(df)
            
        with tab2:
            annot_files = [l for l in annots_path.rglob('*.txt')]
            display_annots = [f.relative_to(annots_path).as_posix() 
                              for f in annot_files]
            chosen_file = st.selectbox(label="Choose a file", 
                                options=display_annots, key='annot_files')
            st.write('All of these files can be imported into Raven directly.')
            df = pd.read_csv(annots_path.joinpath(chosen_file), sep='\t')
            st.dataframe(df)
    