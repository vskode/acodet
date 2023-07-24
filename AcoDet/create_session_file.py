import yaml
import json

def create_session_file():
    with open('simple_config.yml', 'r') as f:
        simple = yaml.safe_load(f)
        
    with open('advanced_config.yml', 'r') as f:
        advanced = yaml.safe_load(f)
        
    session = {**simple, **advanced}

    with open('AcoDet/files/tmp_session.json', 'w') as f:
        json.dump(session, f)