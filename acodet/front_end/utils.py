import streamlit as st
from pathlib import Path
import json

def open_folder_dialogue(path, key):
    try:
        foldernames_list = [x.stem for x in Path(path).iterdir()]

        # create selectbox with the foldernames
        chosen_folder = st.selectbox(label="Choose a folder", 
                                     options=foldernames_list,
                                     key = key)

        # set the full path to be able to refer to it
        directory_path =  Path(path).joinpath(chosen_folder)
        return str(directory_path)
        
    except FileNotFoundError:
        st.write('Folder does not exist, retrying.')
        
def next_button(id, **kwargs):
    if f'b{id}' not in st.session_state:
        setattr(st.session_state, f'b{id}', False)
    val = st.button('Next', key=f'button_{id}', **kwargs)
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

def write_to_session_file(key, value):
    with open('acodet/src/tmp_session.json', 'r') as f:
        session = json.load(f)
    session[key] = value
    with open('acodet/src/tmp_session.json', 'w') as f:
        json.dump(session, f)

def validate_float(input):
    try:
        return float(input)
    except ValueError:
        st.write('<p style="color:red; font-size:10px;">The value you entered is not a number.</p>',
                 unsafe_allow_html=True)

def validate_int(input):
    try:
        return int(input)
    except ValueError:
        st.write('<p style="color:red; font-size:12px;">The value you entered is not a number.</p>',
                 unsafe_allow_html=True)

def make_nested_btn_false_if_dropdown_changed(run_id, preset_id, btn_id):
    with open('acodet/src/tmp_session.json', 'r') as f:
        session = json.load(f)
    if not (session['run_config'] == run_id and \
        session['predefined_settings'] == preset_id):
            setattr(st.session_state, f'b{btn_id}', False)
        