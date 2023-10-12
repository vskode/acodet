RUN_OPTION = "Choose what option to run (currently only Inference is supported)."
SELECT_PRESET = "Based on the preset, different computations will run."
SAMPLE_RATE = """
If you need to change this, make sure that the sample rate 
you set is below the sample rate of your audio files
"""

ENTER_PATH = """
Enter the path to the directory containing the dataset(s) (so one above it). 
The dropdown below will then allow you to choose the dataset(s) you would like to annotate on.
"""

CHOOSE_FOLDER = """
Choose the folder containing the dataset(s) you would like to annotate on.
"""
THRESHOLD = "The threshold will be used to filter the predictions."
ANNOTATIONS_DEFAULT_LOCATION = """
The annotations are stored in this folder by default. If you want to specify another location, choose "No".
"""
CHOOSE_TIMESTAMP_FOLDER = """
These are the time stamps corresponding to computations that 
have been performed on the machine previously. They all contain annotations files
and can be used to filter the annotations with different thresholds or to generate
hourly predictions.
"""

LIMIT = """
Choose between Simple and Sequence limit. Simple limit will only count the
annotations that are above the threshold. Sequence limit will only count the
annotations that are above the threshold and exceed the limit within 20
consecutive sections.
"""

ANNOT_FILES_DROPDOWN = """
Choose the annotation file you would like to inspect.
"""

SC_LIMIT = """
The limit will be used to filter the predictions. The limit is the number of
vocalizations within 20 consecutive sections that need to exceed the threshold.
Higher limits mean less false positives but more false negatives. 
Play around withit and see how the plot changes. 
The idea behind this is to be able to tune the sensitivity of the model
to the noise environment within the dataset.
"""