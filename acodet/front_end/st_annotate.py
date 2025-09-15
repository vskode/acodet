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
            "What would you like to run?",
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

def model_dropdown(key):
    from bacpipe import supported_models, models_needing_checkpoint
    models_without_checkpoint = [m for m in supported_models 
                                 if not m in models_needing_checkpoint]
    st.header("Which model would you like to use?")
    rad = st.radio("Models requiring checkpoints?", 
                   ["Yes", "No"], 
                   horizontal=True, 
                   key=key+'rad',
                   help=help_strings.MODEL_SELECT)
    if rad == 'Yes':
        model = st.selectbox(
            "Models requiring checkpoints:",
            models_needing_checkpoint,
            index=None,
            key=key+'ckpt',
            help=help_strings.MODEL_CHECKPOINTS
        )
    else:
        model = st.selectbox(
            "Models requiring no checkpoints:",
            models_without_checkpoint,
            index=None,
            key=key+'no_ckpt',
            help=help_strings.MODEL_NO_CHECKPOINTS
        )
    rad = True if rad == 'Yes' else False
    utils.write_to_session_file('bool_bacpipe_chckpts', rad)
    if model and not model == 'hbdet':
        utils.write_to_session_file('ModelClassName', 'BacpipeModel')
        utils.write_to_session_file('multiclass', True)
    return model

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
                "Model threshold:", "0.6", help=help_strings.THRESHOLD
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
            
            if st.session_state.ModelClassName == 'BacpipeModel':
                self.bacpipe_model_settings()

    def bacpipe_model_settings(self):
        batch_size = int(utils.user_input(
            "Inference batch size", 
            '8', 
            key='bs', 
            help=help_strings.BATCH_SIZE
            ))
        
        device = st.radio(
            'Device to run computation on',
            options=['cpu', 'cuda', 'auto'],
            index=2,
            key='dv',
            help=help_strings.DEVICE,
            horizontal=True
            )
        
        bool_lin_clfier = st.radio(
            'Would you like to use a pretrained linear classifer '
            'on top of the chosen feature extractor to predict classes?',
            options=['Yes', 'No'],
            index=1,
            key='linclfb',
            horizontal=True
            )
        bool_lin_clfier = True if bool_lin_clfier == 'Yes' else False
        
        if st.session_state.bool_bacpipe_chckpts:
            bacpipe_chckpt_dir = utils.user_input(
                'Folder containing the model checkpoint', 
                '../model_checkpoints',
                key='b_chck',
                help=help_strings.BACPIPE_CHCKPT_DIR
                )
            if not Path(bacpipe_chckpt_dir).exists():
                st.write("Folder does not exist, retrying.")
        else:
            bacpipe_chckpt_dir = ''
        
        if bool_lin_clfier:
            lin_clfier_dir = utils.user_input(
                'Path to linear classifer', 
                '../linear_classifer',
                key='lin_clf',
                help=help_strings.LIN_CLFIER_DIR
                )
            if (
                not Path(lin_clfier_dir).exists() 
                and len([d for d in Path(lin_clfier_dir).glob('*.pt')]) > 0
                ):
                st.write("Path does not exist, retrying.")
        else:
            lin_clfier_dir = ''
        
        utils.write_to_session_file('bacpipe_chckpt_dir', bacpipe_chckpt_dir)
        utils.write_to_session_file('lin_clfier_dir', lin_clfier_dir)
        utils.write_to_session_file('batch_size', batch_size)
        utils.write_to_session_file('device', device)
        utils.write_to_session_file('bool_lin_clfier', bool_lin_clfier)

    def ask_for_multiple_datasets(self):
        st.radio(
            "Would you like to process multiple datasets in this session?",
            (False, True),
            key=f"multi_datasets_{self.key}",
            horizontal=True,
            help=help_strings.MULTI_DATA,
        )


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
    if preset_option in [0, 1]:
        model_select = model_dropdown(key)
        st.session_state.model_name = model_select
    else:
        ml = st.radio('Was this a multiclass classification?', 
                 options=[True, False],
                 key=f'multiclass_{key}')
        utils.write_to_session_file('multiclass', ml)

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
