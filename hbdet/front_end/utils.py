import streamlit as st
import hbdet.global_config as conf
from pathlib import Path


def open_folder_dialogue(path, key):
    try:
        foldernames_list = [x.stem for x in Path(path).iterdir()]

        # create selectbox with the foldernames
        chosen_folder = st.selectbox(label="Choose a folder", 
                                     options=foldernames_list,
                                     key = key)

        # set the full path to be able to refer to it
        directory_path =  Path(path).joinpath(chosen_folder)
        conf.SOUND_FILES_SOURCE = directory_path
        
    except FileNotFoundError:
        st.write('Folder does not exist, retrying.')