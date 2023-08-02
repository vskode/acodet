import streamlit as st
from acodet.front_end import utils


def annotate_options(key='annot'):
    preset_option = int(st.selectbox(
        'What predefined Settings would you like to run?',
        ('1 - generate new annotations',
         '2 - filter existing annotations with different threshold',
         '3 - generate hourly predictions (simple limit and sequence criterion)',
         '4 - generate hourly predictions (only simple limit)',
         '0 - all of the above'), 
        key=key)[0])
    utils.make_nested_btn_false_if_dropdown_changed(1, preset_option, 1)
    utils.next_button(id=1)
    if not st.session_state.b1:
        pass
    else:
        config = dict()
        config['predefined_settings'] = preset_option
        
        if preset_option == 1 or preset_option == 0:
            path = st.text_input("Enter the path to your sound data:", '.')
            config['sound_files_source'] = utils.open_folder_dialogue(path, key='folder_' + key)
            config['thresh'] = utils.validate_float(utils.user_input("Model threshold:", "0.9"))
                
            st.markdown("## Aggregation metrics parameters for hourly presence and hourly counts.")
            st.markdown('### Specify parameters for the simple limit.')
            config['simple_limit'] = utils.validate_int(utils.user_input("Number of annotations for simple limit:", "15"))
                
            st.markdown('### Specify parameters for the sequence limit.')
            config['sc_limit'] = utils.validate_int(utils.user_input("Number of annotations for sequence limit:", "20"))
            config['sc_thresh'] = utils.validate_float(utils.user_input("Threshold for sequence limit:", "0.9"))

        elif preset_option == 2:
            path = st.text_input("Enter the path to your annotation data:", '.')
            config['generated_annotation_source'] = utils.open_folder_dialogue(path, key='folder_' + key)
            config['thresh'] = utils.validate_float(utils.user_input("Rerun annotations Model threshold:", "0.9"))
            
        else:
            st.markdown("## Aggregation metrics parameters for hourly presence and hourly counts.")
            st.markdown('### Specify parameters for the simple limit.')
            
            config['simple_limit'] = utils.validate_int(utils.user_input("Number of annotations for simple limit:", "15"))
            config['thresh'] = utils.validate_float(utils.user_input("Threshold for simple limit:", "0.9"))
                
            st.markdown('### Specify parameters for the sequence limit.')
            config['sc_limit'] = utils.validate_int(utils.user_input("Number of annotations for sequence limit:", "20"))
            config['sc_thresh'] = utils.validate_float(utils.user_input("Threshold for sequence limit:", "0.9"))

        for k, v in config.items():
            utils.write_to_session_file(k, v)
        return True
