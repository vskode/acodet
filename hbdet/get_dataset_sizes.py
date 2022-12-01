import json
from pathlib import Path 
import global_config as conf

TFRECORDS_DIR = list(Path(conf.TFREC_DESTINATION).iterdir())

for path in TFRECORDS_DIR:
    for file in path.glob('**/*json'):
        res = json.load(open(file))
        print(res['dataset']['noise'])