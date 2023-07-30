# acodet - **H**ump**b**ack Whale **Det**ector

<!-- ## Features:
## · Generate raven annotation tables
## · Hourly presence spreadsheets or visualizations
## · Generate new training data
## · Train models -->

- [Installation Instructions for Windows](#installation-instructions-for-windows)
    - [Preliminary software installations](#preliminary-software-installations)
    - [Installation instructions](#installation-instructions)
- [acodet Usage](#acodet-usage)
    - [Usecase 1: Generating annotations](#usecase-1-generating-annotations)
    - [Usecase 2: Generating new training data](#usecase-2-generating-new-training-data)
    - [Usecase 3: Training](#usecase-3-training)
- [FAQ](#faq)
----------------------------------------------------

# Installation Instructions for Windows
### Preliminary software installations:
- install python 3.8: (standard install, no admin privileges needed)
<https://www.python.org/ftp/python/3.8.0/python-3.8.0-amd64.exe>
- install git bash: (default install)
<https://github.com/git-for-windows/git/releases/download/v2.38.1.windows.1/Git-2.38.1-64-bit.exe>

### Installation instructions
- create project directory in location of your choice
- open git bash in project directory (right click, Git Bash here)
- clone the repository:

`git clone https://github.com/vskode/acodet.git`
- Install virtualenv (copy and paste in Git Bash console):

`$HOME/AppData/Local/Programs/Python/Python38/python -m pip install virtualenv`

- Create a new virtual environment (default name env_acodet can be changed):

 `$HOME/AppData/Local/Programs/Python/Python38/python -m virtualenv env_acodet`

- activate newly created virtual environment (change env_acodet if necessary):

`source env_acodet/Scripts/activate`

- Install required packages:

`pip install -r acodet/requirements.txt`

-------------------------

# acodet Usage
Users only need to change the files **simple_congif.yml** and **advanced_config.yml** to use acodet. Once the config files are changed, users can run the program by running the command `python run.py` inside the **acodet** directory.

## Usecase 1: Generating annotations
To generate annotations:
- open the file **simple_config.yml** in any Editor (default is Notepad). 
- change `run_config` to `1`
- change `predefined_settings` to one of the following:
    - `1` for generating annotations with a threshold of 0.5
    - `2` for generating annotations with a custom threshold
        - specify threshold (**thresh**) value in **simple_config.yml** (defaults to 0.9)
    - `3` for generating hourly counts and presence spreadsheets and visualizations (using the sequence criterion and the simple limit)
        - _simple limit_ and _sequence criterion_ are accumulation metrics aiming to deliver hourly presence information, while filtering out false positives
            - _simple limit_ -> only consider annotations if the number of annotations exceeding the **thresh** value is higher than the value for **simple_limit** in **simple_config.yml** (in a given hour in the dataset)
            - _sequence criterion_ -> only consider annotations if the number of consecutive annotations within **sc_con_win** number of windows exceeding the **sc_thresh** value is higher than **sc_limit** (in a given hour in the dataset)
        - hourly counts gives the number of annotations according to the accumulation metrics
        - hourly presence gives a binary (0 -> no whale; 1 -> whale) corresponding to whether the accumulation metrics are satisfied
    - `4` for generating hourly counts and presence spreadsheets and visualizations (using only the simple limit)
    - or `0` to run all of the above in sequece
- change `sound_files_source` to the top level directory containing the dataset(s) you want to annotate

- once finished, save the **simple_config.yml** file

To start the program:
- activate the virtual environment again:

`source env_acodet/Scripts/activate`

- run the run.py script:

`python acodet/run.py`

## Output

The software will now run thorugh your dataset and gerate annotations for every (readable) soundifle within the dataset. While running, a spreadsheet, called stats.csv is continuously updated showing information on the annotations for every file (do not open while program is still running, because the program wont be able to access it).

The program will create a directory called `generated_annotatoins` in the project directory. It will then create a directory corresponding to the date and time that you started the annotation process. Within that directory you will find a directory `thresh_0.5` corresponding to all annotations with a threshold of 0.5. Furthermore you will find the `stats.csv` spreadsheet.

If you have chosen option 2 (or 0) you will also find a directory `thresh_0.x` where the x stands for the custom threshold you specified in the **simple_config.yml** file. Within the `thresh` directories you will find the name of your dataset. 

If you have chosen option 3, 4 or 0 you will find a directory `analysis` within the dataset directory. In that directory you will find spreadsheets for hourly presence and hourly counts, as well as visualizations of the hourly presence and hourly counts.

## Usecase 2: Generating new training data

Either use manually created annotations -> option 2, or create new annotations by reviewing the automatically generated annotations -> option 1.

For option 1, use Raven to open sound files alongside their automatically generated annotations. Edit the column `Predictions/Comments` by writing `n` for noise, `c` for call, or `u` for undefined. If the majority of the shown windows are calls, add the suffix `_allcalls` before the `.txt` ending so that the program will automatically label all of the windows as calls, unless specified as `n`, `c`, or `u`. The suffix `_allnoise` will do the same for noise. The suffix `_annotated` will label all unchanged windows as undefined - thereby essentially ignoring them for the created dataset.

Once finished, insert the top-level directory path to the `reviewed_annotation_source` variable in **simple_config.yml**. 

To generate new training data:
- open the file **simple_config.yml** in any Editor (default is Notepad). 
- change `run_config` to `2`
- change `predefined_settings` to one of the following:
    - `1` for generating training data from reviewed annotations
    - `2` for generating training data from manually created training data (space in between annotations will be interpretted as noise)
- change `sound_files_source` to the top level directory containing the dataset(s) containing the sound files

- once finished, save the **simple_config.yml** file

To start the program:
- activate the virtual environment again:

`source env_acodet/Scripts/activate`

- run the run.py script:

`python acodet/run.py`

## Usecase 3: Training

To train the model:
- open the file **simple_config.yml** in any Editor (default is Notepad). 
- change `run_config` to `3`
- change `predefined_settings` to one of the following:
    - `1` for generating training data from reviewed annotations

- once finished, save the **simple_config.yml** file
- more adcanced changes for model parameters can be done in **advanced_config.yml**

To start the program:
- activate the virtual environment again:

`source env_acodet/Scripts/activate`

- run the run.py script:

`python acodet/run.py`

# FAQ

At the moment installation on the Apple M1 and M2 processors still produce installation errors, this is a known issue and will hopefully be fixed soon.