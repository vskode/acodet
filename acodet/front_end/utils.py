import streamlit as st
import acodet.global_config as conf
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
        
def next_button(id):
    if f'b{id}' not in st.session_state:
        setattr(st.session_state, f'b{id}', False)
    val = st.button('Next', key=f'button_{id}')
    if val:
        setattr(st.session_state, f'b{id}', True)
        
def user_input(label, val, **input_params):           
    c1, c2 = st.columns(2)
    c1.markdown("##")
    input_params.setdefault("key", label)
    c1.markdown(label)
    return c2.text_input('empty', val, label_visibility='hidden', 
                         **input_params)

def user_dropdown(label, vals, **input_params):           
    c1, c2 = st.columns(2)
    c1.markdown("##")
    input_params.setdefault("key", label)
    c1.markdown(label)
    return c2.selectbox('empty', vals, label_visibility='hidden', 
                        **input_params)                      
