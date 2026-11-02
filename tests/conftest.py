"""
Shared setup of the acodet smoke tests.

acodet reads its session file when `global_config` is first imported, and uses
paths relative to the working directory (`acodet/src/tmp_session.json`,
`../trainings`, ...). So before anything from acodet is imported, a throwaway
working directory is created and entered. This way the tests:

- never overwrite the real `acodet/src/tmp_session.json`, and never write to
  the real `../trainings` or `../generated_annotations` folders
- don't depend on the paths in the local config files
- can run at the same time on a shared cluster file system
- behave the same regardless of where pytest is started from

Everything runs on the CPU, so GPU driver problems don't hide real failures.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_AUDIO = REPO_ROOT / "tests" / "test_datasets" / "audio"
TEST_TABLES = REPO_ROOT / "tests" / "test_datasets" / "tables"

# basic_test.py runs bacpipe.play() on import and test.py is the old test
# script with outdated values, neither of them should be collected
collect_ignore = ["basic_test.py", "test.py"]

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# no display on clusters
os.environ.setdefault("MPLBACKEND", "Agg")

SANDBOX = Path(tempfile.mkdtemp(prefix="acodet-smoke-"))
WORK_DIR = SANDBOX / "work"

SESSION_OVERRIDES = {
    "run_config": 2,
    "predefined_settings": 1,
    "sound_files_source": str(TEST_AUDIO),
    "reviewed_annotation_source": str(TEST_TABLES),
    "generated_annotation_source": str(SANDBOX / "generated_annotations"),
    "generated_annotations_folder": str(SANDBOX / "generated_annotations"),
    "annotation_destination": str(SANDBOX / "combined_annotations"),
    "annotation_destination_folder": "dataset",
    # These settings result in the windows of the test annotations, which were
    # generated with 7755 samples at 2 kHz, i.e. 3.8775 s.
    "sample_rate": 2000,
    "context_window_in_seconds": 3.9,
    "stft_frame_len": 1024,
    "number_of_time_bins": 128,
    "ModelClassName": "HumpBackNorthAtlantic",
    "model_name": "Humpback_20221130",
    "load_ckpt_path": False,
    "batch_size": 4,
    "epochs": 1,
    "steps_per_epoch": 2,
    "time_augs": True,
    "mixup_augs": True,
    "spec_aug": True,
    "thresh": 0.6,
    "save_downsampled_files": False,
    "streamlit": False,
    "device": "cpu",
    "bool_lin_clfier": False,
    "bool_bacpipe_chckpts": False,
}


def _create_work_dir():
    (WORK_DIR / "acodet" / "src").mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / "acodet" / "annotation_mappers.json", WORK_DIR / "acodet"
    )
    session = {}
    for config_file in ("simple_config.yml", "advanced_config.yml"):
        shutil.copy(REPO_ROOT / config_file, WORK_DIR)
        with open(REPO_ROOT / config_file, "r") as f:
            session.update(yaml.safe_load(f))
    session.update(SESSION_OVERRIDES)
    with open(WORK_DIR / "acodet" / "src" / "tmp_session.json", "w") as f:
        json.dump(session, f, indent=2)


_create_work_dir()
sys.path.insert(0, str(REPO_ROOT))
os.chdir(WORK_DIR)


def pytest_unconfigure(config):
    # Windows can't remove the current working directory
    os.chdir(REPO_ROOT)
    shutil.rmtree(SANDBOX, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    """Give every test its own output folders."""
    from acodet import global_config as conf

    monkeypatch.setattr(conf, "ANNOT_DEST", str(tmp_path / "combined"))
    monkeypatch.setattr(conf, "GEN_ANNOTS_DIR", str(tmp_path / "generated"))
    monkeypatch.setattr(conf, "GEN_ANNOT_SRC", str(tmp_path / "generated"))
    return tmp_path
