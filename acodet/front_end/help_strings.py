MODEL_SELECT = (
    """Select the model you would like to use for your session. Remember, the model has classes that it was trained 
    on, if those classes don't exist in your data, the model will not generate meaningful results. If you select a model
    remember that some of them will require you to provide the path to the model checkpoints."""
)
MODEL_CHECKPOINTS = (
    """When you select one of these models, you have to provide the path to the location of the checkpoint. Otherwise
    the model will not run successfully."""
)
MODEL_NO_CHECKPOINTS = (
    """These models are all supported in AcoDet by default. Some of them are being downloaded in the background. You don't
    have to do anything else."""
)

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
ANNOTATIONS_TIMESTAMP_FOLDER = """
Specify custom string to append to timestamp for foldername.\n
Example: 2024-01-27_22-24-23___Custom-Dataset-1"""
ANNOTATIONS_TIMESTAMP_RADIO = """
By default the annotations folder is named according to the timestamp when it was created.
By clicking Yes you can add a custom string to make it more specific.
"""
CHOOSE_TIMESTAMP_FOLDER = """
These are the time stamps corresponding to computations that 
have been performed on the machine previously. They all contain annotations files
and can be used to filter the annotations with different thresholds or to generate
hourly predictions.
"""

MULTI_DATA = """
Are there multiple datasets located in the selected folder and would you
like for all of them to be processed? If so select yes, if not, please only
select the desired folder.
"""

SAVE_SELECTION_BTN = """
By clicking, the selection tables of the chosen datasets will be 
recomputed with the limit settings chosen above and saved in the same location
with a name corresponding to the limit name and threshold."""

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
