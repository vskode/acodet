import zipfile
from pathlib import Path
import hbdet.global_config as conf

for model_path in Path(conf.MODEL_DIR).iterdir():
    if not model_path.suffix == '.zip':
        continue
    else:
        with zipfile.ZipFile(model_path, 'r') as model_zip:
            model_zip.extractall(conf.MODEL_DIR)