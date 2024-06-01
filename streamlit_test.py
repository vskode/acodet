from streamlit.testing.v1 import AppTest
from pathlib import Path
import pandas as pd


class TestStreamlitApp:
    def __init__(self):
        self.at = AppTest.from_file("streamlit_app.py", default_timeout=1200)

    def test_run_config_1(self):
        self.at.run()
        key = "annot"
        self.at.selectbox(key=key).select("1 - generate new annotations").run()
        self.at.button(key="button_1").click().run()
        self.test_generation(key, "test_single_dataset_not_nested")
        self.test_generation(key, "test_single_dataset_nested")
        # self.test_generation(key, "test_multi_dataset_nested")

    def test_generation(self, key, selectbox_input, multi_datasets=False):
        self.at.text_input(key="text_" + key).input(
            "tests/test_files/test_audio_files"
        ).run()
        self.at.selectbox(key="folder_" + key).select(selectbox_input).run()
        self.at.radio(key="radio_timestamp_" + key).set_value("Yes").run()
        self.at.text_input(key="Custom Folder string:").input(
            "_auto_testing"
        ).run()
        if multi_datasets:
            self.at.radio(key="multi_datasets_" + key).set_value("Yes").run()
        self.at.button(key="button_4").click().run()
        self.at.button(key="button_5").click().run()
        generated_annots_folder = self.at.session_state[
            "generated_annotation_source"
        ]
        self.test_created_folders(generated_annots_folder, selectbox_input)
        self.delete_tree(Path(generated_annots_folder))

    def test_created_folders(self, generated_annots_folder, selectbox_input):
        mainfold = Path(generated_annots_folder)
        default_thresh = mainfold.joinpath("thresh_0.5")
        assert mainfold.exists()
        assert mainfold.joinpath("stats.csv").exists()
        assert default_thresh.exists()
        assert default_thresh.joinpath(selectbox_input).exists()
        src_tree = [
            f
            for f in Path(self.at.session_state["sound_files_source"]).glob(
                "**/*"
            )
        ]
        src_rel_to = Path(self.at.session_state["sound_files_source"]).parent
        targ_tree = [
            f for f in default_thresh.joinpath(selectbox_input).glob("**/*")
        ]
        for src, targ in zip(src_tree, targ_tree):
            assert list(src.relative_to(src_rel_to).parents) == list(
                targ.relative_to(default_thresh).parents
            )

        file = [
            f for f in default_thresh.joinpath(selectbox_input).glob("*.txt")
        ]
        df = pd.read_csv(file[0])
        assert len(df) == 15

    def delete_tree(self, path):
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                self.delete_tree(child)
        path.rmdir()


T = TestStreamlitApp()
T.test_run_config_1()
