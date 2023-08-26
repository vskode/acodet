import streamlit as st
from acodet.front_end import utils
from pathlib import Path
from acodet import create_session_file

conf = create_session_file.read_session_file()


def annotate_options(key="annot"):
    preset_option = int(
        st.selectbox(
            "What predefined Settings would you like to run?",
            (
                "1 - generate new annotations",
                "2 - filter existing annotations with different threshold",
                "3 - generate hourly predictions",
                "0 - all of the above",
            ),
            key=key,
        )[0]
    )

    st.session_state.preset_option = preset_option
    utils.make_nested_btn_false_if_dropdown_changed(1, preset_option, 1)
    utils.make_nested_btn_false_if_dropdown_changed(
        run_id=1, preset_id=preset_option, btn_id=4
    )
    utils.next_button(id=1)
    if not st.session_state.b1:
        pass
    else:
        config = dict()
        config["predefined_settings"] = preset_option

        if preset_option == 1 or preset_option == 0:
            path = st.text_input("Enter the path to your sound data:", ".")
            config["sound_files_source"] = utils.open_folder_dialogue(
                path, key="folder_" + key
            )
            config["thresh"] = utils.validate_float(
                utils.user_input("Model threshold:", "0.9")
            )

        elif preset_option in [2, 3]:
            default_path = st.radio(
                f"The annotations I would like to filter are located in `{Path(conf['generated_annotations_folder']).resolve()}`:",
                ("Yes", "No"),
                key="radio_" + key,
                horizontal=True,
            )
            if default_path == "Yes":
                config[
                    "generated_annotation_source"
                ] = utils.open_folder_dialogue(
                    key="folder_default_" + key,
                    label="From the timestamps folders, choose the one you would like to work on.",
                    filter_existing_annotations=True,
                )
            elif default_path == "No":
                path = st.text_input(
                    "Enter the path to your annotation data:", "."
                )
                config[
                    "generated_annotation_source"
                ] = utils.open_folder_dialogue(
                    path,
                    key="folder_" + key,
                    label="From the timestamps folders, choose the one you would like to work on.",
                    filter_existing_annotations=True,
                )
                if (
                    Path(config["generated_annotation_source"]).stem
                    + Path(config["generated_annotation_source"]).suffix
                    == f"thresh_{conf['default_threshold']}"
                ):
                    st.write(
                        """
                            <p style="color:red; font-size:14px;">
                            Please choose the top-level folder (usually a timestamp) instead.
                            </p>""",
                        unsafe_allow_html=True,
                    )
            if preset_option == 2:
                config["thresh"] = utils.validate_float(
                    utils.user_input(
                        "Rerun annotations Model threshold:", "0.9"
                    )
                )

        for k, v in config.items():
            utils.write_to_session_file(k, v)
        return True
