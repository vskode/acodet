import streamlit as st
from pathlib import Path
from acodet import create_session_file
from . import help_strings

conf = create_session_file.read_session_file()
import json
import keras


def open_folder_dialogue(
    path=conf["generated_annotations_folder"],
    key="folder",
    label="Choose a folder",
    filter_existing_annotations=False,
    **kwargs,
):
    try:
        if not filter_existing_annotations:
            foldernames_list = [
                f"{x.stem}{x.suffix}"
                for x in Path(path).iterdir()
                if x.is_dir()
            ]
            if f"thresh_{conf['default_threshold']}" in foldernames_list:
                foldernames_list = [f"thresh_{conf['default_threshold']}"]
        else:
            foldernames_list = [
                f"{x.stem}{x.suffix}"
                for x in Path(path).iterdir()
                if x.is_dir()
            ]
            foldernames_list.sort()
            foldernames_list.reverse()
        # create selectbox with the foldernames
        chosen_folder = st.selectbox(
            label=label, options=foldernames_list, key=key, **kwargs
        )

        # set the full path to be able to refer to it
        directory_path = Path(path).joinpath(chosen_folder)
        return str(directory_path)

    except FileNotFoundError:
        st.write("Folder does not exist, retrying.")


def next_button(id, text="Next", **kwargs):
    if f"b{id}" not in st.session_state:
        setattr(st.session_state, f"b{id}", False)
    val = st.button(text, key=f"button_{id}", **kwargs)
    if val:
        setattr(st.session_state, f"b{id}", True)
        make_nested_btns_false_on_click(id)
        if text in ["Run computations", "Next"]:
            st.session_state.run_finished = False


def user_input(label, val, **input_params):
    c1, c2 = st.columns(2)
    c1.markdown("##")
    input_params.setdefault("key", label)
    c1.markdown(label)
    return c2.text_input(" ", val, **input_params)


def user_dropdown(label, vals, **input_params):
    c1, c2 = st.columns(2)
    c1.markdown("##")
    input_params.setdefault("key", label)
    c1.markdown(label)
    return c2.selectbox(" ", vals, **input_params)


def write_to_session_file(key, value):
    if "session_started" in st.session_state:
        setattr(st.session_state, key, value)
    else:
        with open("acodet/src/tmp_session.json", "r") as f:
            session = json.load(f)
        session[key] = value
        with open("acodet/src/tmp_session.json", "w") as f:
            json.dump(session, f)


def validate_float(input):
    try:
        return float(input)
    except ValueError:
        st.write(
            '<p style="color:red; font-size:10px;">The value you entered is not a number.</p>',
            unsafe_allow_html=True,
        )


def validate_int(input):
    try:
        return int(input)
    except ValueError:
        st.write(
            '<p style="color:red; font-size:12px;">The value you entered is not a number.</p>',
            unsafe_allow_html=True,
        )


def make_nested_btn_false_if_dropdown_changed(run_id, preset_id, btn_id):
    if "session_started" in st.session_state:
        session = {**st.session_state}
    else:
        with open("acodet/src/tmp_session.json", "r") as f:
            session = json.load(f)
    if not (
        session["run_config"] == run_id
        and session["predefined_settings"] == preset_id
    ):
        setattr(st.session_state, f"b{btn_id}", False)


def make_nested_btns_false_on_click(btn_id):
    btns = [i for i in range(btn_id + 1, 6)]
    for btn in btns:
        setattr(st.session_state, f"b{btn}", False)


def prepare_run():
    if st.session_state.run_option == 1:
        st.markdown("""---""")
        st.markdown("## Computation started, please wait.")
        if st.session_state.preset_option in [0, 1]:
            if not st.session_state.ModelClassName == 'BacpipeModel':
                kwargs = {
                    "callbacks": TFPredictProgressBar,
                    "progbar1": st.progress(0, text="Current file"),
                    "progbar2": st.progress(0, text="Overall progress"),
                }
            else:
                kwargs = {
                    "progbar1": st.progress(0, text="Current file"),
                    "progbar2": st.progress(0, text="Overall progress"),
                }
        else:
            kwargs = {"progbar1": st.progress(0, text="Progress")}
    return kwargs


class Limits:
    def __init__(self, limit, key):
        """
        A simple class to contain all methods revolving around the limit sliders
        for simple and sequence limit.

        Parameters
        ----------
        limit : string
            either simple or sequence limit, from radio btn
        key : string
            unique identifier for streamlit options
        """
        self.key = "limit_" + key
        self.save_btn = False
        if limit == "Simple limit":
            self.limit_label = "simple_limit"
            self.thresh_label = "thresh"
            self.sc = False
            self.limit_max = 50
        elif limit == "Sequence limit":
            self.limit_label = "sequence_limit"
            self.thresh_label = "sequence_thresh"
            self.sc = True
            self.limit_max = 20

    def create_limit_sliders(self):
        """
        Show sliders for simple and sequence limit, depending on the selection
        of the radio btn.
        """
        self.thresh = st.slider(
            "Threshold",
            0.35,
            0.99,
            conf[self.thresh_label],
            0.01,
            key=f"thresh_slider_{self.key}",
            help=help_strings.THRESHOLD,
        )

        if self.sc:
            self.limit = st.slider(
                "Limit",
                1,
                self.limit_max,
                conf[self.limit_label],
                1,
                key=f"limit_slider_{self.key}",
                help=help_strings.SC_LIMIT,
            )

    def show_save_selection_tables_btn(self):
        """Show save selection tables btn."""
        self.save_btn = st.button(
            "Save tables", self.key, help=help_strings.SAVE_SELECTION_BTN
        )

    def save_selection_tables_with_limit_settings(self):
        """
        Save the selection tables of the chosen dataset again with
        the selected settings of the respective limit.
        """
        self.show_save_selection_tables_btn()
        if self.save_btn:
            st.session_state.progbar_update = st.progress(0, text="Progress")
            write_to_session_file(self.thresh_label, self.thresh)
            if self.sc:
                write_to_session_file(self.limit_label, self.limit)

            import run

            run.main(
                dont_save_plot=True,
                sc=self.sc,
                fetch_config_again=True,
                preset=3,
                save_filtered_selection_tables=True,
            )


class TFPredictProgressBar(keras.callbacks.Callback):
    def __init__(self, num_of_files, progbar1, progbar2, **kwargs):
        self.num_of_files = num_of_files
        self.pr_bar1 = progbar1
        self.pr_bar2 = progbar2

    def on_predict_end(self, logs=None):
        self.pr_bar2.progress(
            st.session_state.progbar1 / self.num_of_files,
            text="Overall progress",
        )

    def on_predict_batch_begin(self, batch, logs=None):
        if self.params["steps"] == 1:
            denominator = 1
        else:
            denominator = self.params["steps"] - 1
        self.pr_bar1.progress(batch / denominator, text="Current file")
