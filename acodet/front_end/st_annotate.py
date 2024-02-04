import streamlit as st
from acodet.front_end import utils
from pathlib import Path
from acodet import create_session_file
from acodet.front_end import help_strings


session_config = create_session_file.read_session_file()


def initial_dropdown(key):
    """
    Show dropdown menu to select what preset to run.

    Parameters
    ----------
    key : string
        unique identifier for streamlit objects

    Returns
    -------
    int
        preset number
    """
    return int(
        st.selectbox(
            "What predefined Settings would you like to run?",
            (
                "1 - generate new annotations",
                "2 - filter existing annotations with different threshold",
                "3 - generate hourly predictions",
                "0 - all of the above",
            ),
            key=key,
            help=help_strings.SELECT_PRESET,
        )[0]
    )


class PresetInterfaceSettings:
    def __init__(self, config, key) -> None:
        """
        Preset settings class. All methods are relevant for displaying
        streamlit objects that will then be used to run the computations.

        Parameters
        ----------
        config : dict
            config dictionsary
        key : string
            unique identifier
        """
        self.config = config
        self.key = key

    def custom_timestamp_dialog(self):
        """
        Show radio buttons asking for custom folder names. If Yes is selected
        allow user input that will be appended to timestamp for uses to give
        custom names to annotation sessions.
        """
        timestamp_radio = st.radio(
            f"Would you like to customize the annotaitons folder?",
            ("No", "Yes"),
            key="radio_" + self.key,
            horizontal=True,
            help=help_strings.ANNOTATIONS_TIMESTAMP_RADIO,
        )
        if timestamp_radio == "Yes":
            self.config["annots_timestamp_folder"] = "___" + utils.user_input(
                "Custom Folder string:",
                "",
                help=help_strings.ANNOTATIONS_TIMESTAMP_FOLDER,
            )
        elif timestamp_radio == "No":
            self.config["annots_timestamp_folder"] = ""

    def ask_to_continue_incomplete_inference(self):
        continue_radio = st.radio(
            f"Would you like to continue a cancelled session?",
            ("No", "Yes"),
            key="radio_continue_session_" + self.key,
            horizontal=True,
            help=help_strings.ANNOTATIONS_TIMESTAMP_RADIO,
        )
        if continue_radio == "Yes":
            past_sessions = list(
                Path(session_config["generated_annotations_folder"]).rglob(
                    Path(self.config["sound_files_source"]).stem
                )
            )
            if len(past_sessions) == 0:
                st.text(
                    f"""Sorry, but no annotations have been found in 
                        `{session_config['generated_annotations_folder']}` on the currently
                        selected dataset (`{self.config['sound_files_source']}`)."""
                )
            else:
                previous_session = st.selectbox(
                    "Which previous session would you like to continue?",
                    past_sessions,
                    key="continue_session_" + self.key,
                    help=help_strings.SELECT_PRESET,
                )
                self.config["timestamp_folder"] = previous_session
        else:
            return True

    def perform_inference(self):
        """
        Interface options when inference is selected, i.e. preset options 0 or 1.
        """
        path = st.text_input(
            "Enter the path to your sound data:",
            "tests/test_files",
            help=help_strings.ENTER_PATH,
        )
        self.config["sound_files_source"] = utils.open_folder_dialogue(
            path, key="folder_" + self.key, help=help_strings.CHOOSE_FOLDER
        )
        self.config["thresh"] = utils.validate_float(
            utils.user_input(
                "Model threshold:", "0.9", help=help_strings.THRESHOLD
            )
        )
        self.advanced_settings()

    def advanced_settings(self):
        """
        Expandable section showing advanced settings options.
        """
        with st.expander(r"**Advanced Settings**"):
            continue_session = self.ask_to_continue_incomplete_inference()

            if continue_session:
                self.custom_timestamp_dialog()

            self.ask_for_multiple_datasets()

    def ask_for_multiple_datasets(self):
        multiple_datasets = st.radio(
            "Would you like to process multiple datasets in this session?",
            ("No", "Yes"),
            key=f"multi_datasets_{self.key}",
            horizontal=True,
            help=help_strings.MULTI_DATA,
        )
        if multiple_datasets == "Yes":
            self.config["multi_datasets"] = True

    def rerun_annotations(self):
        """
        Show options for rerunning annotations and saving the
        selection tables with a different limit.
        """
        self.select_annotation_source_directory()
        self.limit = st.radio(
            "What limit would you like to set?",
            ("Simple limit", "Sequence limit"),
            key=f"limit_selec_{self.key}",
            help=help_strings.LIMIT,
        )

        self.lim_obj = utils.Limits(self.limit, self.key)
        self.lim_obj.create_limit_sliders()
        self.lim_obj.save_selection_tables_with_limit_settings()

    def select_annotation_source_directory(self):
        """
        Streamlit objects for preset options 2 and 3.
        """
        default_path = st.radio(
            f"""The annotations I would like to filter are located in 
            `{Path(session_config['generated_annotations_folder']).resolve()}`:""",
            ("Yes", "No"),
            key="radio_" + self.key,
            horizontal=True,
            help=help_strings.ANNOTATIONS_DEFAULT_LOCATION,
        )
        if default_path == "Yes":
            self.config[
                "generated_annotation_source"
            ] = utils.open_folder_dialogue(
                key="folder_default_" + self.key,
                label="From the timestamps folders, choose the one you would like to work on.",
                help=help_strings.CHOOSE_TIMESTAMP_FOLDER,
                filter_existing_annotations=True,
            )
        elif default_path == "No":
            path = st.text_input(
                "Enter the path to your annotation data:",
                "tests/test_files",
                help=help_strings.ENTER_PATH,
            )
            self.config[
                "generated_annotation_source"
            ] = utils.open_folder_dialogue(
                path,
                key="folder_" + self.key,
                label="From the timestamps folders, choose the one you would like to work on.",
                help=help_strings.CHOOSE_TIMESTAMP_FOLDER,
                filter_existing_annotations=True,
            )
            if (
                Path(self.config["generated_annotation_source"]).stem
                + Path(self.config["generated_annotation_source"]).suffix
                == f"thresh_{session_config['default_threshold']}"
            ):
                st.write(
                    """
                        <p style="color:red; font-size:14px;">
                        Please choose the top-level folder (usually a timestamp) instead.
                        </p>""",
                    unsafe_allow_html=True,
                )


def annotate_options(key="annot"):
    """
    Caller function for inference settings. Calls all necessary components
    to show streamlit objects where users can choose what settings to run.

    Parameters
    ----------
    key : str, optional
        unique identifier for streamlit objects, by default "annot"

    Returns
    -------
    boolean
        True once all settings have been entered
    """
    preset_option = initial_dropdown(key)

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
        interface_settings = PresetInterfaceSettings(config, key)

        if preset_option == 1 or preset_option == 0:
            interface_settings.perform_inference()

        elif preset_option == 2:
            interface_settings.rerun_annotations()

        elif preset_option == 3:
            interface_settings.select_annotation_source_directory()
            interface_settings.ask_for_multiple_datasets()

        for k, v in interface_settings.config.items():
            utils.write_to_session_file(k, v)
        return True
