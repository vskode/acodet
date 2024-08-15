import os, sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

########### MODIFY SESSION SETTINGS BEFORE GLOBAL CONFIG IS IMPORTED #########
from acodet.create_session_file import create_session_file

create_session_file()
import json

with open("acodet/src/tmp_session.json", "r") as f:
    session = json.load(f)
session["sound_files_source"] = "tests/test_files/test_audio_files"
session[
    "generated_annotation_source"
] = "tests/test_files/test_generated_annotations"
session[
    "annotation_destination"
] = "tests/test_files/test_combined_annotations"
session[
    "generated_annotations_folder"
] = "tests/test_files/test_generated_annotations"

session[
    "reviewed_annotation_source"
] = "tests/test_files/test_generated_annotations"
session["tfrecords_destination_folder"] = "tests/test_files/test_tfrecords"

with open("acodet/src/tmp_session.json", "w") as f:
    json.dump(session, f)
##############################################################################


from acodet.annotate import run_annotation, filter_annots_by_thresh
from acodet.funcs import return_windowed_file, get_train_set_size
from acodet.models import GoogleMod
from acodet.combine_annotations import generate_final_annotations
from acodet.tfrec import write_tfrec_dataset
from acodet.train import run_training
from acodet import global_config as conf



class TestDetection(unittest.TestCase):
    def test_annotation(self):
        self.time_stamp = run_annotation()
        df = pd.read_csv(
            (
                Path(conf.GEN_ANNOTS_DIR)
                .joinpath(self.time_stamp)
                .joinpath("stats.csv")
            )
        )
        self.assertEqual(
            df["number of predictions with thresh>0.8"][0],
            326,
            "Number of predictions is not what it should be.",
        )

        filter_annots_by_thresh(self.time_stamp)
        file = list(
            Path(conf.GEN_ANNOT_SRC)
            .joinpath(self.time_stamp)
            .joinpath(f"thresh_{conf.THRESH}")
            .glob("**/*.txt")
        )[0]
        df = pd.read_csv(file)
        self.assertEqual(
            len(df),
            309,
            "Number of predictions from filtered thresholds " "is incorrect.",
        )


class TestTraining(unittest.TestCase):
    def test_model_load(self):
        model = GoogleMod(load_g_ckpt=False).model
        self.assertGreater(len(model.layers), 15)

    # def test_tfrecord_loading(self):
    #     data_dir = list(Path(conf.TFREC_DESTINATION).iterdir())
    #     n_train, n_noise = get_train_set_size(data_dir)
    #     self.assertEqual(n_train, 517)
    #     self.assertEqual(n_noise, 42)

class TestTFRecordCreation(unittest.TestCase):
    def test_tfrecord(self):
        time_stamp = list(Path(conf.ANNOT_DEST).iterdir())[-1]
        write_tfrec_dataset(annot_dir=time_stamp, active_learning=False)
        metadata_file_path = Path(conf.TFREC_DESTINATION).joinpath(
            "dataset_meta_train.json"
        )
        self.assertEqual(
            metadata_file_path.exists(),
            1,
            "TFRecords metadata file was not created.",
        )

        with open(metadata_file_path, "r") as f:
            data = json.load(f)
            self.assertEqual(
                data["dataset"]["size"]["train"],
                517,
                "TFRecords files has wrong number of datapoints.",
            )

    def test_combined_annotation(self):
        generate_final_annotations(active_learning=False)
        time_stamp = list(Path(conf.GEN_ANNOTS_DIR).iterdir())[-1].stem
        combined_annots_path = (
            Path(conf.ANNOT_DEST)
            .joinpath(time_stamp)
            .joinpath("combined_annotations.csv")
        )
        self.assertEqual(
            combined_annots_path.exists(),
            1,
            "csv file containing combined_annotations does not exist.",
        )
        df = pd.read_csv(combined_annots_path)
        self.assertEqual(
            df.start.iloc[-1],
            1795.2825,
            "The annotations in combined_annotations.csv don't seem to be identical",
        )


if __name__ == "__main__":
    unittest.main()
