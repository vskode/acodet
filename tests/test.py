import os, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

########### MODIFY SESSION SETTINGS BEFORE GLOBAL CONFIG IS IMPORTED #########
from acodet.create_session_file import create_session_file

create_session_file()
import json

with open("acodet/src/tmp_session.json", "r") as f:
    session = json.load(f)
session["sound_files_source"] = "tests/test_datasets/audio"
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
session["annotation_destination_folder"] = "tests/test_files/test_dataset"

with open("acodet/src/tmp_session.json", "w") as f:
    json.dump(session, f)
##############################################################################


from acodet.annotate import run_annotation, filter_annots_by_thresh
from acodet.models import GoogleMod
from acodet.combine_annotations import generate_final_annotations
from acodet import global_config as conf



def test_annotation():
    time_stamp = run_annotation()
    df = pd.read_csv(
        (
            Path(conf.GEN_ANNOTS_DIR)
            .joinpath(time_stamp)
            .joinpath("stats.csv")
        )
    )
    assert df["number of predictions with thresh>0.8"].sum() == 908, \
        "Number of predictions is not what it should be."

    filter_annots_by_thresh(time_stamp)
    file = list(
        Path(conf.GEN_ANNOT_SRC)
        .joinpath(time_stamp)
        .joinpath(f"thresh_{conf.THRESH}")
        .glob("**/*.txt")
    )
    file.sort()
    df = pd.read_csv(file[-1])
    assert len(df) == 250, \
        "Number of predictions from filtered thresholds is incorrect."


def test_model_load():
    model = GoogleMod(load_g_ckpt=False).model
    assert len(model.layers) == 26, "Model has wrong number of layers."


def test_combined_annotation():
    generate_final_annotations(active_learning=False)
    time_stamp = list(Path(conf.GEN_ANNOTS_DIR).iterdir())[-1].stem
    combined_annots_path = (
        Path(conf.ANNOT_DEST)
        .joinpath(time_stamp)
        .joinpath("combined_annotations.csv")
    )
    assert combined_annots_path.exists() == 1, \
        "csv file containing combined_annotations does not exist."
    df = pd.read_csv(combined_annots_path)
    assert df.start.iloc[-1] == 1709.9775, \
        "The annotations in combined_annotations.csv don't seem to be identical"

# test_annotation()
# test_model_load()
# test_combined_annotation()