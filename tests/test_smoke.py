"""
Smoke tests for acodet, using the audio and tables in tests/test_datasets.

Run them from the repository root, on every machine and environment:

    python -m pytest tests/test_smoke.py -v -rs

No downloads are needed and everything runs on the CPU. Tests whose
dependencies (e.g. tensorflow or bacpipe) are not installed in the current
environment are skipped, `-rs` lists them with the reason. Tests of real models
are opt-in, because they download checkpoints:

    ACODET_TEST_REAL_MODEL=1              default humpback detector
    ACODET_TEST_BACPIPE_MODEL=birdnet     a bacpipe model with a classifier
"""
import importlib
import importlib.util
import os
import re
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO_ROOT, TEST_AUDIO, TEST_TABLES

from acodet import combine_annotations as ca
from acodet import funcs
from acodet import global_config as conf

WINDOW_S = 7755 / 2000
TABLES = sorted(TEST_TABLES.glob("**/*.txt"))
SAMANA_DIR = TEST_AUDIO / "BERCHOK_SAMANA_200901_4"
AUDIO_FILES = sorted(SAMANA_DIR.glob("*.wav"))


############################## helper functions ##############################


def sound_stem(table):
    return table.stem.split("_annot")[0]


def raw_table(table):
    """One row per selection, the way Raven displays it."""
    return (
        pd.read_csv(table, sep="\t")
        .drop_duplicates(subset="Selection")
        .reset_index(drop=True)
    )


def flags_of(df):
    return df[conf.ANNOTATION_COLUMN].astype(str).str.strip().str.lower()


def n_windows(df):
    """
    Number of windows the annotations are expected to be split into. Written
    out independently of the implementation.
    """
    total = 0
    for duration in df["End Time (s)"] - df["Begin Time (s)"]:
        n_full = int(duration // WINDOW_S)
        remainder = duration - n_full * WINDOW_S
        total += max(1, n_full + (n_full >= 1 and remainder >= WINDOW_S / 2))
    return total


def read_output():
    dest = Path(conf.ANNOT_DEST)
    combined = pd.read_csv(dest / "combined_annotations.csv")
    noise = pd.read_csv(dest / "explicit_noise.csv")
    for df in (combined, noise):
        df["stem"] = [Path(f).stem for f in df["filename"]]
    return combined, noise


def overlaps_annotation(row, annotations, eps=1e-6):
    return (
        (row.start < annotations["End Time (s)"] - eps)
        & (row.end > annotations["Begin Time (s)"] + eps)
    ).any()


def annotation_df(rows, **columns):
    """rows: list of (begin, end, comment)"""
    df = pd.DataFrame(
        rows, columns=["Begin Time (s)", "End Time (s)", conf.ANNOTATION_COLUMN]
    )
    df["Selection"] = np.arange(1, len(df) + 1)
    df["Low Freq (Hz)"] = 50.0
    df["High Freq (Hz)"] = 1000.0
    for name, values in columns.items():
        df[name.replace("_", " ")] = values
    return df


################################ configuration ###############################


def test_context_window_fits_test_annotations():
    assert conf.CONTEXT_WIN == 7755
    assert ca.get_window_duration_s() == pytest.approx(WINDOW_S)


def test_repository_configs_contain_every_required_key():
    source = (REPO_ROOT / "acodet" / "global_config.py").read_text()
    required = set(re.findall(r"""session\[["'](\w+)["']\]""", source))
    available = {}
    for config_file in ("simple_config.yml", "advanced_config.yml"):
        with open(REPO_ROOT / config_file, "r") as f:
            available.update(yaml.safe_load(f))
    assert required, "No session keys found in global_config.py"
    assert not required - set(available)


def test_removed_tfrecord_code_stays_removed():
    for module in ("tfrec", "torch_eval", "split_daily_annots"):
        assert importlib.util.find_spec(f"acodet.{module}") is None
    sources = list((REPO_ROOT / "acodet").glob("**/*.py"))
    sources.append(REPO_ROOT / "run.py")
    pattern = re.compile(
        r"^\s*(from|import)\s+\S*(tfrec|torch_eval|split_daily_annots|tensorflow_io|keras_cv)\b",
        re.MULTILINE,
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert not pattern.search(text), f"Removed module imported in {source}"


def test_pyproject_only_adds_to_bacpipe():
    import tomllib

    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        dependencies = tomllib.load(f)["project"]["dependencies"]
    assert sorted(dependencies) == ["bacpipe==1.3.5", "gdown", "streamlit"]


# module: packages it can't be imported without
MODULES = {
    "acodet.funcs": [],
    "acodet.combine_annotations": [],
    "acodet.create_session_file": [],
    "acodet.hourly_presence": ["seaborn"],
    "acodet.torch_augment": ["torch", "torchaudio"],
    "acodet.torch_data": ["torch"],
    "acodet.torch_train": ["torch", "torchaudio"],
    "acodet.torchmodels": ["torch", "torchaudio", "timm"],
    "acodet.models": ["torch", "torchaudio", "timm"],
    "acodet.annotate": ["torch", "torchaudio", "timm"],
    "acodet.train": [],
    "acodet.evaluate": ["torch", "torchaudio", "timm", "sklearn"],
    "acodet.tf_dataloader": ["tensorflow"],
    "acodet.plot_utils": ["tensorflow", "seaborn"],
    "acodet.transfer_weights": ["tensorflow"],
    "acodet.augmentation": ["tensorflow", "seaborn"],
    "acodet.front_end.utils": ["streamlit"],
    "acodet.front_end.st_annotate": ["streamlit"],
    "acodet.front_end.st_generate_data": ["streamlit"],
    "acodet.front_end.st_train": ["streamlit"],
    "acodet.front_end.st_visualization": ["streamlit"],
}


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    for package in MODULES[module]:
        pytest.importorskip(package)
    importlib.import_module(module)


def test_tf220_layers_import_with_keras_3():
    pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    if int(keras.__version__.split(".")[0]) < 3:
        pytest.skip("acodet.tf220 is only used with Keras 3")
    importlib.import_module("acodet.tf220")


###################### RUN_CONFIG 2: labels of real tables ###################


def test_test_data_is_complete():
    assert len(TABLES) == 10
    for table in TABLES:
        assert list(TEST_AUDIO.glob(f"**/{sound_stem(table)}.wav")), table


def test_active_learning_labels_match_review_flags():
    """RUN_CONFIG 2, preset 1"""
    ca.generate_final_annotations(active_learning=True)
    combined, noise = read_output()

    for table in TABLES:
        raw = raw_table(table)
        flags = flags_of(raw)
        scores = pd.to_numeric(raw[conf.ANNOTATION_COLUMN], errors="coerce")
        if "_allcalls" in table.stem:
            calls = raw[~flags.isin(["n", "u"])]
            explicit = raw[flags == "n"]
            plain_noise = raw.iloc[:0]
        else:
            calls = raw[flags == "c"]
            explicit = raw[(flags == "n") | (scores > 0.9)]
            plain_noise = raw[~flags.isin(["c", "n", "u"]) & ~(scores > 0.9)]

        stem = sound_stem(table)
        got = combined[combined.stem == stem]
        got_noise = noise[noise.stem == stem]
        assert (got.label == 1).sum() == n_windows(calls), table.name
        assert (got.label == 0).sum() == n_windows(plain_noise), table.name
        assert len(got_noise) == n_windows(explicit), table.name

    assert set(combined.label.unique()) <= {0, 1}
    assert set(noise.label.unique()) <= {0}


def test_every_output_row_belongs_to_a_real_sound_file():
    ca.generate_final_annotations(active_learning=True)
    combined, noise = read_output()
    all_rows = pd.concat([combined, noise])
    assert len(all_rows) > 0
    assert all(Path(f).is_file() for f in all_rows.filename.unique())
    assert set(all_rows.stem) <= {sound_stem(t) for t in TABLES}
    assert (all_rows.start >= 0).all()
    assert (all_rows.end > all_rows.start).all()
    # long annotations in these tables were split into windows
    assert (all_rows.end - all_rows.start).max() < WINDOW_S * 1.5


@pytest.mark.parametrize(
    "table_name, expected_windows",
    [
        ("Samana_PU194_20090111_133000_annot_2022-11-24_17_allcalls.txt", 15),
        ("Samana_PU194_20090112_052000_annot_2022-11-24_17_allcalls.txt", 10),
    ],
)
def test_long_noise_annotations_are_split_into_explicit_noise(
    table_name, expected_windows, tmp_path
):
    # These tables contain one 56.25 s and one 38.86 s annotation flagged 'n'.
    # Both are converted by hand: 56.25 / 3.8775 = 14.51 -> 14 windows plus one
    # aligned to the end, 38.86 / 3.8775 = 10.02 -> 10 windows.
    table = next(t for t in TABLES if t.name == table_name)
    raw = raw_table(table)
    long_annot = raw[raw["End Time (s)"] - raw["Begin Time (s)"] > 2 * WINDOW_S]
    assert len(long_annot) == 1

    # only this annotation, windows of other selections might overlap with it
    only_long = tmp_path / table.name
    long_annot.to_csv(only_long, sep="\t", index=False)
    ca.generate_final_annotations(
        annotation_files=[only_long], active_learning=True
    )
    combined, noise = read_output()
    begin = long_annot["Begin Time (s)"].iloc[0]
    end = long_annot["End Time (s)"].iloc[0]
    assert len(combined) == 0
    assert len(noise) == expected_windows
    assert (noise.start >= begin - 1e-6).all()
    assert (noise.end <= end + 1e-6).all()
    assert (noise.end - noise.start).values == pytest.approx(WINDOW_S)


@pytest.mark.parametrize("active_learning", [True, False])
def test_gap_noise_never_overlaps_annotations(active_learning):
    """RUN_CONFIG 2, presets 2 (active_learning=False) and 3"""
    ca.generate_final_annotations(
        active_learning=active_learning, inbetween_noise=True
    )
    combined, _ = read_output()
    n_checked = 0
    for table in TABLES:
        if active_learning and "_allnoise" in table.stem:
            # label 0 is also given to the annotations of these tables
            continue
        raw = raw_table(table)
        windows = combined[
            (combined.stem == sound_stem(table)) & (combined.label == 0)
        ]
        for row in windows.itertuples():
            assert not overlaps_annotation(row, raw), f"{table.name}: {row}"
            assert row.end - row.start == pytest.approx(WINDOW_S)
            n_checked += 1
    assert n_checked > 0, "No noise was added between the annotations"


def test_manual_annotations_use_flags_and_no_filename_suffixes():
    """RUN_CONFIG 2, preset 2 without the gap noise"""
    ca.generate_final_annotations(active_learning=False)
    combined, noise = read_output()
    for table in TABLES:
        raw = raw_table(table)
        flags = flags_of(raw)
        stem = sound_stem(table)
        got = combined[combined.stem == stem]
        assert (got.label == 1).sum() == n_windows(raw[~flags.isin(["n", "u"])])
        assert (noise.stem == stem).sum() == n_windows(raw[flags == "n"])
    assert (combined.label == 1).all()


def test_tables_under_a_test_folder_become_the_test_set():
    # the reviewed annotation source is .../tests/test_datasets/tables
    ca.generate_final_annotations(active_learning=True)
    combined, noise = read_output()
    assert (combined.subset == "test").all()
    if len(noise):
        assert (noise.subset == "test").all()


@pytest.fixture
def train_tables(tmp_path, monkeypatch):
    """The test tables in a folder that is recognised as a training set."""
    folder = tmp_path / "train_tables"
    shutil.copytree(TEST_TABLES, folder)
    monkeypatch.setattr(conf, "REV_ANNOT_SRC", str(folder))
    return folder


def test_validation_files_split_train_and_val(train_tables):
    val_stems = ["Samana_PU194_20090111_003000", "Samana_PU194_20090112_052000"]
    pd.DataFrame({"validation_files": val_stems}).to_csv(
        train_tables / "validation_files.csv", index=False
    )
    ca.generate_final_annotations(active_learning=True)
    combined, noise = read_output()
    for df in (combined, noise):
        assert set(df.subset.unique()) <= {"train", "val"}
        assert set(df[df.subset == "val"].stem) <= set(val_stems)
        assert not set(df[df.subset == "train"].stem) & set(val_stems)

    train, val, test = ca.read_annotations_from_file(conf.ANNOT_DEST)
    assert len(test) == 0
    assert len(train) > 0 and len(val) > 0
    assert len(train) + len(val) == len(combined) + len(noise)
    assert not set(train.filename) & set(val.filename)

    # no validation audio in the noise of the noise augmentation
    train_noise = ca.get_train_noise_df(train)
    assert (train_noise.label == 0).all()
    assert not set(train_noise.filename) & set(val.filename)


def test_train_folder_without_validation_file_still_runs(train_tables):
    ca.generate_final_annotations(active_learning=True)
    combined, _ = read_output()
    assert len(combined) > 0


def test_files_without_active_learning_suffix_are_ignored(train_tables):
    (train_tables / "notes.txt").write_text("not an annotation table")
    ca.generate_final_annotations(active_learning=True)
    assert len(read_output()[0]) > 0


def test_empty_tables_do_not_crash(train_tables):
    (train_tables / "empty_annot_x_allcalls.txt").write_text("")
    header = TABLES[0].read_text().splitlines()[0]
    (train_tables / "header_only_annot_x_allcalls.txt").write_text(header + "\n")
    ca.generate_final_annotations(active_learning=True)
    assert len(read_output()[0]) > 0


def test_table_without_sound_file_is_skipped(train_tables):
    shutil.copy(TABLES[0], train_tables / "doesnotexist_annot_x_allnoise.txt")
    ca.generate_final_annotations(active_learning=True)
    combined, noise = read_output()
    assert "empty" not in set(combined.filename) | set(noise.filename)
    assert all(Path(f).is_file() for f in combined.filename.unique())


def test_folder_without_tables_gives_clear_error(tmp_path, monkeypatch):
    empty = tmp_path / "no_tables"
    empty.mkdir()
    monkeypatch.setattr(conf, "REV_ANNOT_SRC", str(empty))
    with pytest.raises(ValueError, match="No .txt files"):
        ca.generate_final_annotations(active_learning=True)


################### RUN_CONFIG 2: labels of constructed tables ###############


def test_undefined_section_never_becomes_noise():
    df = annotation_df([(0, 3.8775, "c"), (7.755, 15.51, "u"), (30, 33.8775, "c")])
    out = ca.get_labels(
        Path("rec_annot_x_allcalls.txt"),
        df,
        active_learning=True,
        inbetween_noise=True,
        clip_duration=WINDOW_S,
    )
    undefined = df.iloc[[1]]
    noise = out[out.label == ca.NOISE]
    assert len(noise) > 0
    for begin, end in zip(noise["Begin Time (s)"], noise["End Time (s)"]):
        row = SimpleNamespace(start=begin, end=end)
        assert not overlaps_annotation(row, undefined)
    assert not (flags_of(out) == "u").any()


def test_flags_are_found_when_first_value_is_empty():
    df = annotation_df(
        [(0, 3, np.nan), (5, 8, "c"), (10, 13, "n"), (15, 18, "u"), (20, 23, 0.5)]
    )
    out = ca.get_labels(
        Path("rec_annot_x_allnoise.txt"),
        df,
        active_learning=True,
        clip_duration=WINDOW_S,
    )
    labels = dict(zip(out["Begin Time (s)"], out["label"]))
    assert labels == {
        0: ca.NOISE,
        5: ca.CALL,
        10: ca.EXPLICIT_NOISE,
        20: ca.NOISE,
    }


def test_high_score_in_noise_file_is_explicit_noise():
    df = annotation_df([(0, 3, 0.95), (5, 8, 0.5)])
    out = ca.get_labels(
        Path("rec_annot_x_allnoise.txt"),
        df,
        active_learning=True,
        clip_duration=WINDOW_S,
    )
    assert list(out["label"]) == [ca.EXPLICIT_NOISE, ca.NOISE]


def test_annotated_file_without_flag_is_dropped():
    df = annotation_df([(0, 3, 0.7), (5, 8, "c")])
    out = ca.get_labels(
        Path("rec_annot_x_annotated.txt"),
        df,
        active_learning=True,
        clip_duration=WINDOW_S,
    )
    assert list(out["Begin Time (s)"]) == [5]
    assert list(out["label"]) == [ca.CALL]


def test_raven_views_are_not_counted_twice():
    df = annotation_df([(0, 3, "c"), (0, 3, "c"), (5, 8, "c")])
    df["Selection"] = [1, 1, 2]
    out = ca.get_labels(Path("rec.txt"), df, clip_duration=WINDOW_S)
    assert len(out) == 2


def test_gaps_start_after_the_longest_previous_annotation():
    # the first annotation ends after the second one
    df = annotation_df([(0, 12, "c"), (1, 2, "c"), (20, 23, "c")])
    out = ca.add_inbetween_noise(
        ca.label_manual_annotations(df, Path("rec.txt"), window_s=100),
        WINDOW_S,
    )
    noise = out[out.label == ca.NOISE]
    assert noise["Begin Time (s)"].min() == pytest.approx(12)
    assert noise["End Time (s)"].max() <= 20


def test_gaps_are_not_filled_across_sound_files():
    df = annotation_df(
        [(0, 3, "c"), (50, 53, "c")], Begin_File=["a.wav", "b.wav"]
    )
    out = ca.add_inbetween_noise(
        ca.label_manual_annotations(df, Path("rec.txt"), window_s=100),
        WINDOW_S,
    )
    assert (out.label != ca.NOISE).all()


def test_gap_noise_gets_its_own_file_offset():
    df = annotation_df(
        [(100, 103, "c"), (120, 123, "c")], **{"File Offset (s)": [10.0, 30.0]}
    )
    out = ca.add_inbetween_noise(
        ca.label_manual_annotations(df, Path("rec.txt"), window_s=100),
        WINDOW_S,
    )
    noise = out[out.label == ca.NOISE]
    assert len(noise) > 0
    offset = noise["File Offset (s)"] - noise["Begin Time (s)"]
    assert offset.values == pytest.approx(-90)
    assert noise["File Offset (s)"].min() == pytest.approx(13)


def test_gap_noise_does_not_cover_the_end_of_a_split_annotation():
    # 2.1 windows: the last 0.1 windows of the call are in none of the windows
    call_end = 2.1 * WINDOW_S
    df = annotation_df([(0, call_end, "c"), (call_end + 20, call_end + 23, "c")])
    out = ca.get_labels(
        Path("rec.txt"), df, inbetween_noise=True, clip_duration=WINDOW_S
    )
    noise = out[out.label == ca.NOISE]
    assert len(out[out.label == ca.CALL]) == 3
    assert len(noise) > 0
    assert noise["Begin Time (s)"].min() == pytest.approx(call_end)


def test_long_annotation_is_split_without_losing_rows():
    df = annotation_df(
        [(100, 110, "c"), (200, 203, "c")], **{"File Offset (s)": [5.0, 105.0]}
    )
    out = ca.seperate_long_annotations(df, window_s=WINDOW_S)
    pieces = out[out["Begin Time (s)"] < 150]
    # 10 / 3.8775 = 2.58: two windows and one aligned to the end
    assert len(pieces) == 3
    assert (pieces["End Time (s)"] <= 110 + 1e-6).all()
    assert pieces["End Time (s)"].max() == pytest.approx(110)
    assert pieces["File Offset (s)"].nunique() == 3
    assert (out["Begin Time (s)"] == 200).sum() == 1


def test_standardize_uses_file_offset(tmp_path, monkeypatch):
    table = tmp_path / "rec_annot_x_allcalls.txt"
    annotation_df([(100, 103, "c")], **{"File Offset (s)": [5.0]}).to_csv(
        table, sep="\t", index=False
    )
    sound_dir = tmp_path / "sound"
    sound_dir.mkdir()
    (sound_dir / "rec.wav").touch()
    monkeypatch.setattr(conf, "SOUND_FILES_SOURCE", str(sound_dir))
    calls, _ = ca.finalize_annotation(
        table, active_learning=True, clip_duration=WINDOW_S
    )
    assert calls.start.iloc[0] == 5
    assert calls.end.iloc[0] == 8


@pytest.mark.parametrize(
    "parts, expected",
    [
        (("data", "ilaria_training_dataset", "tables", "Tables_testing"), "test"),
        (("data", "test_runs", "train_tables"), "train"),
        (("data", "training", "tables"), "train"),
        (("data", "tables"), None),
        (("a_train_and_test_folder",), "test"),
    ],
)
def test_subset_is_decided_by_the_closest_folder(parts, expected):
    assert ca.get_dataset_subset_from_path(Path(*parts)) == expected


@pytest.mark.parametrize("n_files, n_test", [(1, 0), (4, 0), (5, 1), (10, 2)])
def test_default_test_subset(n_files, n_test):
    calls = pd.DataFrame({"filename": [f"f{i}.wav" for i in range(n_files)]})
    noise = pd.DataFrame({"filename": [f"f{i}.wav" for i in range(n_files)]})
    calls, noise = ca._assign_default_test_subset([calls, noise])
    assert (calls.subset == "test").sum() == n_test
    assert list(calls.subset) == list(noise.subset)


def test_sound_file_with_longer_name_is_not_matched(tmp_path, monkeypatch):
    sound_dir = tmp_path / "sound"
    sound_dir.mkdir()
    for name in ("rec_1.wav", "rec_10.wav", "rec_1_annot_x_allcalls.txt"):
        (sound_dir / name).touch()
    monkeypatch.setattr(conf, "SOUND_FILES_SOURCE", str(sound_dir))
    match = ca.get_corresponding_sound_file(Path("rec_1_annot_x_allcalls.txt"))
    assert Path(match).name == "rec_1.wav"
    assert ca.get_corresponding_sound_file(Path("other_annot_x.txt")) is None


def test_paths_with_special_characters_are_found(tmp_path, monkeypatch):
    # e.g. "Extreme SSD/marine mammals [2024]"
    sound_dir = tmp_path / "Extreme SSD" / "marine mammals [2024]"
    sound_dir.mkdir(parents=True)
    (sound_dir / "rec.wav").touch()
    monkeypatch.setattr(conf, "SOUND_FILES_SOURCE", str(sound_dir))
    match = ca.get_corresponding_sound_file(Path("rec_annot_x_allcalls.txt"))
    assert Path(match).name == "rec.wav"


def test_bacpipe_window_length_with_and_without_bacpipe(monkeypatch):
    monkeypatch.setattr(conf, "MODELCLASSNAME", "BacpipeModel")
    monkeypatch.setattr(
        ca, "get_bacpipe_model_constants", lambda name: (48000, 144000)
    )
    assert ca.get_window_duration_s() == 3

    def no_bacpipe(name):
        raise ImportError("No module named bacpipe")

    monkeypatch.setattr(ca, "get_bacpipe_model_constants", no_bacpipe)
    assert ca.get_window_duration_s() == pytest.approx(WINDOW_S)


######################## data loaders: labels and audio ######################


@pytest.fixture
def small_dataset(monkeypatch, tmp_path):
    """
    Combined annotations from real audio with all three subsets, with sizes
    that fit the batch size of 4: 12 train (4 of them explicit noise),
    8 val and 12 test windows.
    """
    def rows(file, n, subset, first=0, label=None):
        return pd.DataFrame(
            {
                "filename": str(file),
                "Selection": np.arange(n) + 1,
                "label": label if label is not None else np.arange(n) % 2,
                "start": (np.arange(n) + first) * WINDOW_S,
                "end": (np.arange(n) + first + 1) * WINDOW_S,
                "freq_min": 50.0,
                "freq_max": 1000.0,
                "subset": subset,
            }
        )

    dest = tmp_path / "small_dataset"
    dest.mkdir()
    pd.concat(
        [
            rows(AUDIO_FILES[0], 8, "train"),
            rows(AUDIO_FILES[1], 8, "val"),
            rows(AUDIO_FILES[2], 12, "test"),
        ]
    ).to_csv(dest / "combined_annotations.csv", index=False)
    rows(AUDIO_FILES[0], 4, "train", first=8, label=0).to_csv(
        dest / "explicit_noise.csv", index=False
    )
    monkeypatch.setattr(conf, "ANNOT_DEST", str(dest))
    return dest


@pytest.fixture
def torch_loaders(small_dataset, monkeypatch):
    pytest.importorskip("torch")
    from acodet import torch_data

    # one real worker process, this also tests the spawned workers on Windows
    monkeypatch.setattr(torch_data, "NUM_CORES", 1)
    return torch_data.Loader(str(small_dataset))


def test_split_sizes_of_written_dataset(small_dataset):
    train, val, test = ca.read_annotations_from_file(str(small_dataset))
    assert (len(train), len(val), len(test)) == (12, 8, 12)
    assert (train.source == "explicit_noise").sum() == 4


def test_noise_augmentation_only_uses_training_data(small_dataset):
    train, val, test = ca.read_annotations_from_file(str(small_dataset))
    noise = ca.get_train_noise_df(train)
    assert len(noise) == 4
    assert set(noise.filename) == {str(AUDIO_FILES[0])}


def test_one_window_annotation_is_one_frame(torch_loaders):
    # float rounding of end - start used to load many windows twice
    assert len(torch_loaders.train) == 12
    assert len(torch_loaders.val) == 8
    assert len(torch_loaders.test) == 12


def test_rounding_of_window_length_gives_one_frame():
    pytest.importorskip("torch")
    from acodet.torch_data import AudioDataset

    frame = conf.CONTEXT_WIN / conf.SR
    starts = np.arange(500) * frame + 11.6325
    df = pd.DataFrame(
        {
            "filename": "rec.wav",
            "label": 1,
            "start": starts,
            "end": starts + frame,
        }
    )
    assert len(AudioDataset(df)) == 500


@pytest.mark.parametrize("loader_name", ["train", "val", "test"])
def test_torch_loader_returns_correct_audio_and_labels(
    torch_loaders, small_dataset, loader_name
):
    import torch

    expected = pd.concat(
        [
            pd.read_csv(small_dataset / "combined_annotations.csv"),
            pd.read_csv(small_dataset / "explicit_noise.csv"),
        ]
    )
    expected = expected[expected.subset == loader_name]
    loader = getattr(torch_loaders, f"{loader_name}_loader")()

    waves, labels, paths, starts = next(iter(loader))
    assert waves.shape == (conf.BATCH_SIZE, conf.CONTEXT_WIN)
    assert labels.dtype == torch.float32
    assert torch.isfinite(waves).all()
    for label, path, start in zip(labels, paths, starts):
        row = expected[
            (expected.filename == path)
            & np.isclose(expected.start, start.item(), atol=1e-3)
        ]
        assert len(row) == 1
        assert row.label.iloc[0] == label.item()


def test_torch_loader_audio_is_the_annotated_section(torch_loaders):
    import librosa as lb

    item = torch_loaders.test[3]
    audio, _ = lb.load(
        item["path"], sr=conf.SR, offset=item["start"], duration=WINDOW_S
    )
    n = min(len(audio), conf.CONTEXT_WIN)
    assert item["wave"].numpy()[:n] == pytest.approx(audio[:n])


def test_torch_noise_loader_runs(torch_loaders):
    waves, labels, _, _ = next(iter(torch_loaders.noise_loader()))
    assert waves.shape == (conf.BATCH_SIZE, conf.CONTEXT_WIN)
    assert (labels == 0).all()


def test_short_last_frame_is_filled_to_full_length():
    pytest.importorskip("torch")
    from acodet.torch_data import AudioDataset

    # the last window starts just before the end of the 600 s sound file
    df = pd.DataFrame(
        {
            "filename": [str(AUDIO_FILES[0])],
            "label": [1],
            "start": [599.0],
            "end": [599.0 + WINDOW_S],
        }
    )
    assert len(AudioDataset(df)[0]["wave"]) == conf.CONTEXT_WIN


def test_tf_loader_returns_correct_audio_and_labels(small_dataset):
    tf = pytest.importorskip("tensorflow")
    from acodet.tf_dataloader import TFLoader

    loader = TFLoader(str(small_dataset))
    assert (loader.n_train, loader.n_val, loader.n_eval, loader.n_noise) == (
        12, 8, 12, 4,
    )
    expected = ca.read_annotations_from_file(str(small_dataset))[2]
    samples = list(loader.test_loader().take(4))
    for sample, label in zip(samples, expected.label):
        audio, got_label = sample
        assert audio.shape == (conf.CONTEXT_WIN,)
        assert audio.dtype == tf.float32
        assert int(got_label) == int(label)


def test_tf_spectrogram_pipeline(small_dataset):
    pytest.importorskip("tensorflow")
    from acodet.tf_dataloader import TFLoader, make_spec_tensor

    loader = TFLoader(str(small_dataset))
    spec, label = next(iter(make_spec_tensor(loader.train_loader())))
    assert tuple(spec.shape) == (128, 64)
    assert int(label) in (0, 1)


def test_tf_augmentation_pipeline(small_dataset):
    pytest.importorskip("tensorflow")
    pytest.importorskip("seaborn")
    from acodet.augmentation import run_augment_pipeline
    from acodet.tf_dataloader import TFLoader, make_spec_tensor

    loader = TFLoader(str(small_dataset))
    data = run_augment_pipeline(
        make_spec_tensor(loader.train_loader()),
        make_spec_tensor(loader.noise_loader()),
        loader.n_noise,
        loader.n_train,
        time_augs=True,
        mixup_augs=True,
        spec_aug=True,
    )
    spec, _ = next(iter(data))
    assert tuple(spec.shape) == (128, 64)


############################### augmentation #################################


@pytest.mark.parametrize(
    "time_augs, mixup_augs, spec_aug",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_torch_augmentation_flags(monkeypatch, time_augs, mixup_augs, spec_aug):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from acodet.torch_augment import TorchAugment

    monkeypatch.setattr(conf, "TIME_AUGS", time_augs)
    monkeypatch.setattr(conf, "MIXUP_AUGS", mixup_augs)
    monkeypatch.setattr(conf, "SPEC_AUG", spec_aug)
    augment = TorchAugment(64, 128)
    assert augment.do_timeshift == time_augs
    assert augment.do_noiseaug == mixup_augs
    assert augment.do_specaug == spec_aug

    x = torch.rand(4, 1, 64, 128)
    out = augment(x, torch.rand(4, 64, 128))
    assert out.shape == (4 * (1 + time_augs + mixup_augs + spec_aug), 1, 64, 128)


def test_mixup_handles_smaller_last_batch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from acodet.torch_augment import NoiseAugment

    out = NoiseAugment()(torch.rand(3, 64, 128), torch.rand(4, 64, 128))
    assert out.shape == (3, 64, 128)


######################## RUN_CONFIG 1: annotation ############################


class StubModel:
    """Offline replacement of the keras model, 0.95 and 0.2 take turns."""

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, windows, training=False, callbacks=None):
        scores = np.full([len(windows), 1], 0.2, dtype=np.float32)
        scores[::2] = 0.95
        return scores


def analysis_files(timestamp):
    return list(Path(conf.GEN_ANNOTS_DIR, timestamp).glob("**/*.csv"))


def test_inference_keeps_predictions_of_all_batches(monkeypatch):
    pytest.importorskip("tensorflow")
    file = AUDIO_FILES[0]
    n_samples = len(funcs.load_audio(file))
    assert n_samples == 600 * conf.SR

    monkeypatch.setattr(conf, "PRED_BATCH_SIZE", 50 * conf.CONTEXT_WIN)
    preds = funcs.run_inference(file, StubModel())
    assert len(preds) == int(np.ceil(n_samples / conf.CONTEXT_WIN))


def test_all_annotation_steps_with_stub_model(monkeypatch):
    """RUN_CONFIG 1, preset 0: annotate, filter and hourly presence"""
    pytest.importorskip("tensorflow")
    pytest.importorskip("timm")
    pytest.importorskip("seaborn")
    import run
    from acodet import models

    monkeypatch.setattr(models, "init_model", lambda **kwargs: StubModel())
    monkeypatch.setattr(conf, "RUN_CONFIG", 1)
    monkeypatch.setattr(conf, "PRESET", 0)
    timestamp = run.main()

    out_dir = Path(conf.GEN_ANNOTS_DIR, timestamp)
    stats = pd.read_csv(out_dir / "stats.csv")
    assert len(stats) == 10
    assert (stats["number of predictions"] > 0).all()
    # half of the windows are above the threshold, in all batches of the file
    n_windows_in_file = int(np.ceil(600 * conf.SR / conf.CONTEXT_WIN))
    samana = stats[stats.filename.astype(str).str.contains("Samana")]
    assert (
        samana["number of predictions"] == int(np.ceil(n_windows_in_file / 2))
    ).all()

    tables = list((out_dir / conf.THRESH_LABEL).glob("**/*.txt"))
    assert len(tables) == 10
    table = pd.read_csv(tables[0], sep="\t")
    assert (table[conf.ANNOTATION_COLUMN] > conf.DEFAULT_THRESH).all()
    assert (table["End Time (s)"] - table["Begin Time (s)"]).values == (
        pytest.approx(WINDOW_S)
    )

    assert list((out_dir / f"thresh_{conf.THRESH}").glob("**/*.txt"))
    names = {f.name for f in analysis_files(timestamp)}
    assert "hourly_presence_simple_limit.csv" in names
    assert "hourly_presence_sequence_limit.csv" in names


def test_annotations_can_be_reused_as_training_data(monkeypatch, tmp_path):
    """Generated annotations -> reviewed annotations -> dataset"""
    pytest.importorskip("tensorflow")
    pytest.importorskip("timm")
    from acodet import annotate, models

    monkeypatch.setattr(models, "init_model", lambda **kwargs: StubModel())
    timestamp = annotate.run_annotation()

    reviewed = tmp_path / "reviewed"
    reviewed.mkdir()
    generated = Path(conf.GEN_ANNOTS_DIR, timestamp).glob("**/*.txt")
    for file in generated:
        shutil.copy(file, reviewed / f"{file.stem}_allcalls.txt")
    monkeypatch.setattr(conf, "REV_ANNOT_SRC", str(reviewed))
    monkeypatch.setattr(conf, "ANNOT_DEST", str(tmp_path / "from_generated"))

    ca.generate_final_annotations(active_learning=True)
    combined, _ = read_output()
    assert len(combined) > 0
    assert (combined.label == 1).all()
    assert all(Path(f).is_file() for f in combined.filename.unique())


def test_annotation_of_empty_file_does_not_stop_the_run(monkeypatch, tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("timm")
    from acodet import annotate, models

    sound_dir = tmp_path / "20200101_120000_dir"
    sound_dir.mkdir()
    shutil.copy(AUDIO_FILES[0], sound_dir / "good_20090111_003000.wav")
    (sound_dir / "broken_20090111_013000.wav").write_bytes(b"")
    monkeypatch.setattr(conf, "SOUND_FILES_SOURCE", str(sound_dir))
    monkeypatch.setattr(models, "init_model", lambda **kwargs: StubModel())
    timestamp = annotate.run_annotation(relativ_path=str(sound_dir))
    stats = pd.read_csv(Path(conf.GEN_ANNOTS_DIR, timestamp, "stats.csv"))
    assert len(stats) == 1
    assert "good" in str(stats.filename.iloc[0])


def test_score_filter_handles_mixed_flag_columns():
    df = pd.DataFrame(
        {conf.ANNOTATION_COLUMN: ["n", 0.7, "c", "0.9", "u", "text", " N "]}
    )
    clean = funcs.remove_str_flags_from_predictions(df)
    assert list(clean[conf.ANNOTATION_COLUMN].dropna()) == [0.7, 0.9]
    assert len(clean) == 3


def test_multiclass_tables_contain_the_scores_of_their_time_bin(tmp_path):
    predictions = np.array(
        [
            # time bins:  0    1    2    3    4
            [0.9, 0.1, 0.2, 0.1, 0.6],  # class_a
            [0.1, 0.1, 0.8, 0.7, 0.3],  # class_b
        ]
    )
    funcs.save_combined_and_multiclass_dfs(
        predictions,
        ["class_a", "class_b"],
        lambda folder: tmp_path / folder,
        Path("rec.wav"),
        "model",
    )
    combined = pd.read_csv(
        tmp_path / "All_Combined" / "rec_annot_model_combined.txt", sep="\t"
    )
    # time bin 1 is below the threshold of 0.5
    assert list(combined.Selection) == [1, 3, 4, 5]
    assert list(combined.class_a) == pytest.approx([0.9, 0.2, 0.1, 0.6])
    assert list(combined.class_b) == pytest.approx([0.1, 0.8, 0.7, 0.3])

    multiclass = pd.read_csv(
        tmp_path / "multiclass" / "rec_annot_model_multiclass.txt", sep="\t"
    )
    assert list(multiclass[conf.ANNOTATION_COLUMN]) == [
        "0.9000__class_a",
        "0.8000__class_b",
        "0.7000__class_b",
        "0.6000__class_a",
    ]


########################### RUN_CONFIG 2 through run.py ######################


@pytest.mark.parametrize("preset", [1, 2, 3])
def test_run_main_generates_dataset(monkeypatch, preset):
    pytest.importorskip("timm")
    import run

    monkeypatch.setattr(conf, "RUN_CONFIG", 2)
    monkeypatch.setattr(conf, "PRESET", preset)
    run.main()
    combined, _ = read_output()
    assert len(combined) > 0
    assert set(combined.label.unique()) <= {0, 1}
    if preset == 2:
        # with manual annotations, label 0 can only come from the gaps
        assert (combined.label == 0).sum() > 0


def test_run_main_accepts_preset_as_argument(monkeypatch):
    pytest.importorskip("timm")
    import run

    monkeypatch.setattr(conf, "RUN_CONFIG", 2)
    run.main(preset=2)
    assert len(read_output()[0]) > 0


def test_removed_tfrecord_preset_gives_clear_error(monkeypatch):
    pytest.importorskip("timm")
    import run

    monkeypatch.setattr(conf, "RUN_CONFIG", 2)
    with pytest.raises(ValueError, match="tfrecords"):
        run.main(preset=4)


########################### RUN_CONFIG 3: training ###########################


@pytest.fixture
def torch_model_run(small_dataset, monkeypatch):
    pytest.importorskip("timm")
    pytest.importorskip("sklearn")
    import run
    from acodet import models, torch_data

    real_init_model = models.init_model
    monkeypatch.setattr(conf, "MODELCLASSNAME", "TorchModel")
    monkeypatch.setattr(conf, "RUN_CONFIG", 3)
    monkeypatch.setattr(torch_data, "NUM_CORES", 1)
    monkeypatch.setattr(
        models,
        "init_model",
        lambda **kwargs: real_init_model(
            model_name="TorchModel", training_path=False
        ),
    )
    return run


def test_torch_model_trains_and_saves(torch_model_run, monkeypatch):
    before = set(Path("..", "trainings").glob("torchmodel_*"))
    monkeypatch.setattr(conf, "PRESET", 1)
    torch_model_run.main()

    new_dirs = set(Path("..", "trainings").glob("torchmodel_*")) - before
    assert len(new_dirs) == 1
    out_dir = new_dirs.pop()
    assert list(out_dir.glob("*.pt"))
    stats = pd.read_csv(out_dir / "training_stats.csv")
    assert len(stats) == conf.EPOCHS
    assert np.isfinite(stats.loss).all()
    assert (out_dir / "global_config.txt").is_file()


def test_torch_model_evaluates(torch_model_run, monkeypatch):
    monkeypatch.setattr(conf, "PRESET", 2)
    torch_model_run.main()
    results = Path("..", "trainings", conf.MODEL_NAME, "evaluation_dataset")
    assert (results / "best_f1_stats.csv").is_file()
    assert (results / "roc_curve.png").is_file()
    # nothing is written next to the audio files
    assert not list(TEST_AUDIO.glob("**/*_prediction.txt"))


def keep_windows(dataset, subset, n):
    """Only keep the first n windows of a subset in combined_annotations.csv."""
    path = dataset / "combined_annotations.csv"
    df = pd.read_csv(path)
    pd.concat(
        [df[df.subset != subset], df[df.subset == subset].head(n)]
    ).to_csv(path, index=False)


def test_torch_training_with_last_batches_of_one_window(
    torch_model_run, small_dataset, monkeypatch
):
    monkeypatch.setattr(conf, "BATCH_SIZE", 4)
    # 5 + 4 explicit noise = 9 train windows and 5 val windows
    keep_windows(small_dataset, "train", 5)
    keep_windows(small_dataset, "val", 5)
    monkeypatch.setattr(conf, "PRESET", 1)
    torch_model_run.main()
    stats = list(Path("..", "trainings").glob("torchmodel_*/training_stats.csv"))
    assert stats
    assert np.isfinite(pd.read_csv(stats[-1]).loss).all()


@pytest.mark.parametrize(
    "n_windows",
    [
        1,  # a single window, of a single class
        3,  # fewer windows than example predictions are saved
        5,  # the last batch has one window
    ],
)
def test_torch_model_evaluates_few_windows(
    torch_model_run, small_dataset, monkeypatch, n_windows
):
    monkeypatch.setattr(conf, "BATCH_SIZE", 4)
    keep_windows(small_dataset, "test", n_windows)
    monkeypatch.setattr(conf, "PRESET", 2)
    torch_model_run.main()
    results = Path("..", "trainings", conf.MODEL_NAME, "evaluation_dataset")
    assert (results / "best_f1_stats.csv").is_file()
    assert list((results / "example_predictions").glob("*_prediction.txt"))


def test_evaluation_without_test_windows_stops_cleanly(
    torch_model_run, small_dataset, monkeypatch
):
    from acodet.evaluate import evaluate

    keep_windows(small_dataset, "test", 0)
    assert evaluate() == 1


@pytest.mark.parametrize(
    "path",
    [
        "C:\\data\\deep\\rec.1.wav",
        "/data/deep/rec.1.flac",
        "..\\data/rec.1.wav",
        "rec.1.wav",
    ],
)
def test_audio_id_handles_paths_of_all_systems(path):
    pytest.importorskip("timm")
    pytest.importorskip("sklearn")
    from acodet.evaluate import audio_id_from_path

    assert audio_id_from_path(path) == "rec.1"


@pytest.mark.parametrize("n_windows", [1, 3, 4])
def test_pcen_keeps_the_batch_axis(n_windows):
    torch = pytest.importorskip("torch")
    from acodet.humpback_model_dir.torch_PCEN import PCEN

    pcen = PCEN(num_channels=1, alpha=0.98, smooth_coef=0.025, trainable=True)
    output = pcen(torch.rand(n_windows, 1, 64, 128))
    assert tuple(output.shape) == (n_windows, 64, 128)


def test_inference_of_file_with_one_window(monkeypatch):
    pytest.importorskip("timm")
    pytest.importorskip("tensorflow")
    from acodet.torchmodels import TorchModel

    audio = np.random.rand(conf.CONTEXT_WIN).astype("float32")
    monkeypatch.setattr(conf, "MODELCLASSNAME", "TorchModel")
    monkeypatch.setattr(funcs, "load_audio", lambda *args, **kwargs: audio)
    model = TorchModel(effnet="b0").eval()
    predictions = funcs.run_inference("one_window.wav", model)
    assert predictions.shape == (1,)


@pytest.mark.parametrize(
    "parent_dirs",
    [
        Path("rec_CH2", "sub", "dir"),
        PureWindowsPath("rec_CH2\\sub\\dir"),
        PurePosixPath("rec_CH2/sub/dir"),
    ],
)
def test_channel_comes_from_the_top_directory(parent_dirs):
    assert funcs.get_channel(funcs.get_top_dir(parent_dirs)) == 1


def test_channel_of_directory_given_as_string():
    assert funcs.get_top_dir("rec_CH3") == "rec_CH3"
    assert funcs.get_channel(funcs.get_top_dir("rec")) == 0


def test_keras_resnet_builds():
    pytest.importorskip("tensorflow")
    pytest.importorskip("timm")
    import keras

    from acodet.models import GoogleMod

    try:
        model = GoogleMod(load_g_ckpt=False).model
    except Exception as e:
        if int(keras.__version__.split(".")[0]) >= 3:
            pytest.xfail(f"GoogleMod does not build with Keras 3: {e}")
        raise
    assert len(model.layers) == 26


################################## bacpipe ###################################


def test_linear_probe_checkpoint_formats():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from acodet.models import _probe_state_dict_from_checkpoint as to_state_dict

    weights = {"weight": torch.zeros(1, 4), "bias": torch.zeros(1)}
    target = ["probe.weight", "probe.bias"]
    expected = set(target)

    old_names = {f"clfier.{k}": v for k, v in weights.items()}
    new_names = {f"probe.{k}": v for k, v in weights.items()}
    module = torch.nn.Linear(4, 1)

    assert set(to_state_dict(old_names, target)) == expected
    assert set(to_state_dict(new_names, target)) == expected
    assert set(to_state_dict(weights, target)) == expected
    # older acodet versions pickled the bound state_dict method
    assert set(to_state_dict(module.state_dict, target)) == expected
    assert set(to_state_dict(module, target)) == expected


def test_installed_bacpipe_has_the_api_acodet_uses():
    bacpipe = pytest.importorskip("bacpipe")
    from bacpipe.core.audio_processor import _get_model_constants
    from bacpipe.embedding_evaluation.probing import train_probe
    from bacpipe.model_pipelines.runner import Classifier

    for name in (
        "Embedder",
        "ensure_models_exist",
        "settings",
        "TF_MODELS",
        "EMBEDDING_DIMENSIONS",
    ):
        assert hasattr(bacpipe, name), name
    for name in (
        "model_base_path",
        "global_batch_size",
        "run_pretrained_classifier",
    ):
        assert hasattr(bacpipe.settings, name), name
    assert hasattr(train_probe, "LinearProbe")
    assert callable(_get_model_constants)
    assert callable(Classifier.make_classification_dict)
    assert hasattr(bacpipe.Embedder, "get_embeddings_from_model")


def test_bacpipe_constants_without_loading_a_model():
    pytest.importorskip("bacpipe")
    assert funcs.get_bacpipe_model_constants("birdnet") == (48000, 144000)


def test_multiclass_output_with_bacpipe_classification_dict(tmp_path):
    pytest.importorskip("bacpipe")
    classes = ["class_a", "class_b", "class_c"]
    predictions = np.array(
        [[0.9, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.1], [0.7, 0.1, 0.1]]
    )
    model = SimpleNamespace(
        embedder=SimpleNamespace(model=SimpleNamespace(classes=classes))
    )
    funcs.save_multiclass_dfs(
        Path("rec.wav"),
        model,
        lambda folder: tmp_path / folder,
        "model",
        predictions,
    )
    class_a = pd.read_csv(
        tmp_path / "class_a" / "rec_annot_model_class_a.txt", sep="\t"
    )
    assert list(class_a.Selection) == [1, 4]
    assert list(class_a[conf.ANNOTATION_COLUMN]) == pytest.approx([0.9, 0.7])
    # class_c never exceeds the threshold
    assert not (tmp_path / "class_c").exists()
    assert (tmp_path / "All_Combined").is_dir()


################################## streamlit #################################


def test_streamlit_app_starts_without_exception():
    pytest.importorskip("timm")
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(
        str(REPO_ROOT / "streamlit_app.py"), default_timeout=300
    )
    app.run()
    assert not app.exception


############################### opt-in real models ###########################


@pytest.mark.skipif(
    not os.environ.get("ACODET_TEST_REAL_MODEL"),
    reason="Set ACODET_TEST_REAL_MODEL=1 to test the real detector (may download it)",
)
def test_real_humpback_model_annotates(monkeypatch):
    pytest.importorskip("tensorflow")
    pytest.importorskip("timm")
    from acodet import annotate

    monkeypatch.setattr(conf, "MODEL_DIR", str(REPO_ROOT / "acodet/src/models"))
    timestamp = annotate.run_annotation()
    stats = pd.read_csv(Path(conf.GEN_ANNOTS_DIR, timestamp, "stats.csv"))
    assert len(stats) == 10
    assert stats["number of predictions"].sum() > 0
    assert stats["average prediction value"].dropna().between(0, 1).all()


@pytest.mark.skipif(
    not os.environ.get("ACODET_TEST_BACPIPE_MODEL"),
    reason="Set ACODET_TEST_BACPIPE_MODEL=<model> to test a real bacpipe model",
)
def test_real_bacpipe_model_annotates(monkeypatch):
    pytest.importorskip("bacpipe")
    pytest.importorskip("timm")
    from acodet import models

    monkeypatch.setattr(conf, "MODELCLASSNAME", "BacpipeModel")
    monkeypatch.setattr(conf, "MODEL_NAME", os.environ["ACODET_TEST_BACPIPE_MODEL"])
    # both are overwritten with the values of the bacpipe model
    monkeypatch.setattr(conf, "SR", conf.SR)
    monkeypatch.setattr(conf, "CONTEXT_WIN", conf.CONTEXT_WIN)
    model = models.init_model(model_name="BacpipeModel", training_path=False)
    preds = funcs.run_inference(AUDIO_FILES[0], model)
    assert len(preds) > 0
    assert np.isfinite(preds).all()
