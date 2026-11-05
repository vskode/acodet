import yaml
import json
import streamlit as st


def create_session_file(simple_config_file=None, advanced_config_file=None):
    if not simple_config_file:
        simple_config_file = "simple_config.yml"

    if not advanced_config_file:
        advanced_config_file = "advanced_config.yml"

    with open(simple_config_file, "r") as f:
        simple = yaml.safe_load(f)

    with open(advanced_config_file, "r") as f:
        advanced = yaml.safe_load(f)

    session = {**simple, **advanced}

    if "session_started" in st.session_state:
        for k, v in session.items():
            setattr(st.session_state, k, v)
    else:
        with open("acodet/src/tmp_session.json", "w") as f:
            json.dump(session, f, indent=2)


def read_session_file():
    if "session_started" in st.session_state:
        session = {**st.session_state}
    else:
        with open("acodet/src/tmp_session.json", "r") as f:
            session = json.load(f)
    return session
