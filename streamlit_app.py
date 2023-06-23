import streamlit as st
import run
import hbdet.global_config as conf
from pathlib import Path


def annotate_options():
    preset_option = st.selectbox(
        'What predefined Settings would you like to run?',
        ('1 - generate new annotations',
         '2 - filter existing annotations with different threshold',
         '3 - generate hourly predictions (simple limit and sequence criterion)',
         '4 - generate hourly predictions (only simple limit)',
         '0 - all of the above'))
    st.write('You selected:', preset_option)
    preset_option = int(preset_option[0])
    
    if preset_option == 1 or preset_option == 0:
        # f = st.file_uploader("Choose a sound file", 
        #                      type=(["tsv","csv","txt","tab","xlsx","xls", "wav", "aif"]))
        # Folder picker button
        # path = ''
        # while path == '':
        path = st.text_input("Enter the path to your sound data:", '.')
        
        # read subfolders in a give directory based on the actual directory level
        foldernames_list = [x.stem for x in Path(path).iterdir()]

        # create selectbox with the foldernames
        chosen_folder = st.selectbox(label="Choose a folder", options=foldernames_list)

        # set the full path to be able to refer to it
        directory_path =  Path(path).joinpath(chosen_folder)
        conf.SOUND_FILES_SOURCE = directory_path

    return preset_option

def generate_data_options():
    preset_option = st.selectbox(
        'How would you like run the program?',
        ('1 - generate new training data from reviewed annotations',
         '2 - generate new training data from reviewed annotations '
         'and fill space between annotations with noise annotations'))
    st.write('You selected:', preset_option)
    return int(preset_option[0])
    
def train_options():
    preset_option = st.selectbox(   
        'How would you like run the program?',
        ('1 - continue training on existing model and save model in the end',
         '2 - evaluate saved model', 
         '3 - evaluate model checkpoint',
         '4 - save model specified in advanced config'))
    st.write('You selected:', preset_option)
    return int(preset_option[0])















option = st.selectbox(
    'How would you like run the program?',
    ('1 - Annotate', 
     '2 - Generate new training data', 
     '3 - Train'))
st.write('You selected:', option)
option = int(option[0])
conf.RUN_CONFIG = option

if option == 1:
    try:    
        conf.PRESET = annotate_options()
    except:
        st.write('Folder does not exist, retrying.')
elif option == 2:
    conf.PRESET = generate_data_options()
if option == 3:
    conf.PRESET = train_options()


if st.button('Run'):
    st.write('Program started')
    run.main()
else:
    st.write('Waiting on user entry.')
    
    
    
    

