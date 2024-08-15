import streamlit as st
from acodet.front_end import utils


def generate_data_options(key="gen_data"):
    preset_option = int(
        st.selectbox(
            "How would you like run the program?",
            (
                "1 - generate new training data from reviewed annotations",
                "2 - generate new training data from reviewed annotations "
                "and fill space between annotations with noise annotations",
            ),
            key=key,
        )[0]
    )
    st.session_state.preset_option = preset_option
    utils.make_nested_btn_false_if_dropdown_changed(2, preset_option, 2)
    utils.make_nested_btn_false_if_dropdown_changed(
        run_id=1, preset_id=preset_option, btn_id=4
    )
    utils.next_button(id=2)
    if not st.session_state.b2:
        pass
    else:
        config = dict()
        config["predefined_settings"] = preset_option

        if preset_option == 1:
            path1_sound = st.text_input(
                "Enter the path to your sound data:", "."
            )
            config["sound_files_source"] = utils.open_folder_dialogue(
                path1_sound, key="source_folder_" + key
            )
            path2_annots = st.text_input(
                "Enter the path to your reviewed annotations:", "."
            )
            config["reviewed_annotation_source"] = utils.open_folder_dialogue(
                path2_annots, key="reviewed_annotations_folder" + key
            )

            st.markdown("### Settings")
            st.markdown("#### Audio preprocessing")
            config["sample_rate"] = utils.validate_int(
                utils.user_input("sample rate [Hz]", "2000")
            )
            config["context_window_in_seconds"] = utils.validate_float(
                utils.user_input("context window length [s]", "3.9")
            )
            st.markdown("#### Spectrogram settings")
            config["stft_frame_len"] = utils.validate_int(
                utils.user_input("STFT frame length [samples]", "1024")
            )
            config["number_of_time_bins"] = utils.validate_int(
                utils.user_input("number of time bins", "128")
            )
            st.markdown("#### TFRecord creationg settings")
            config["tfrecs_limit_per_file"] = utils.validate_int(
                utils.user_input(
                    "limit of context windows per tfrecord file", "600"
                )
            )
            config["train_ratio"] = utils.validate_float(
                utils.user_input("trainratio", "0.7")
            )
            config["test_val_ratio"] = utils.validate_float(
                utils.user_input("test validation ratio", "0.7")
            )

        elif preset_option == 2:
            path1_sound = st.text_input(
                "Enter the path to your sound data:", "."
            )
            config["sound_files_source"] = utils.open_folder_dialogue(
                path1_sound, key="source_folder_" + key
            )
            path2_annots = st.text_input(
                "Enter the path to your reviewed annotations:", "."
            )
            config["reviewed_annotation_source"] = utils.open_folder_dialogue(
                path2_annots, key="reviewed_annotations_folder" + key
            )

            st.markdown("### Settings")
            st.markdown("#### Audio preprocessing")
            config["sample_rate"] = utils.validate_int(
                utils.user_input("sample rate [Hz]", "2000")
            )
            config["context_window_in_seconds"] = utils.validate_float(
                utils.user_input("context window length [s]", "3.9")
            )
            st.markdown("#### Spectrogram settings")
            config["stft_frame_len"] = utils.validate_int(
                utils.user_input("STFT frame length [samples]", "1024")
            )
            config["number_of_time_bins"] = utils.validate_int(
                utils.user_input("number of time bins", "128")
            )
            st.markdown("#### TFRecord creationg settings")
            config["tfrecs_limit_per_file"] = utils.validate_int(
                utils.user_input(
                    "limit of context windows per tfrecord file", "600"
                )
            )
            config["train_ratio"] = utils.validate_float(
                utils.user_input("trainratio", "0.7")
            )
            config["test_val_ratio"] = utils.validate_float(
                utils.user_input("test validation ratio", "0.7")
            )

        for k, v in config.items():
            utils.write_to_session_file(k, v)
        return True
