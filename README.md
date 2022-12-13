# HBDet - **H**ump**b**ack Whale **Det**ector

## Features:
## · Generate raven annotation tables
## · Hourly presence spreadsheets or visualizations
## · Generate new training data
## · Train models

----------------------------------------------------

# Installation Instructions for Windows
### Necessary software installations:
- install python 3.8: (standard install, no admin privileges needed)
<https://www.python.org/ftp/python/3.8.0/python-3.8.0-amd64.exe>
- install git bash: (default install)
<https://github.com/git-for-windows/git/releases/download/v2.38.1.windows.1/Git-2.38.1-64-bit.exe>

### Set up installation
- create project directory in location of your choice
- open git bash in project directory (right click, Git Bash here)
- clone the repository:

`git clone https://github.com/vskode/hbdet.git`
- Install virtualenv (copy and paste in Git Bash console):

`C:\Users\%username%\AppData\Local\Programs\Python\Python38\python -m pip install virtualenv`

- Create a new virtual environment (default name env_hbdet can be changed):

 `C:\Users\%username%\AppData\Local\Programs\Python\Python38\python -m virtualenv env_hbdet`

- activate newly created virtual environment (change env_hbdet if necessary):

`source env_hbdet/Scripts/activate`

- Install required packages:

`pip install -r hbdet/requirements.txt`

-------------------------

# hbdet Usage
Users only need to change the files **simple_congif.yml** and **advanced_config.yml** to use hbdet. Once the config files are changed, users can run the program by running the command `python run.py` inside the **hbdet** directory.

## Usecase 1: Generating annotations
To generate annotations:
- open the file **simple_config.yml** in any Editor (default is Notepad). 
- change `run_config` to `1`
- change `predefined_settings` to one of the following:
    - 1 for generating annotations with a threshold of 0.5
    - 2 for generating annotations with a custom threshold
    - 4 for generating a hourly predictions spreadsheet and visualization
    - or 0 to run all of the above in sequece
- change `sound_files_source` to the top level directory containing the dataset(s) you want to annotate

- once finished, save the **simple_config.yml** file

To start the program:
- activate the virtual environment again:

`source env_hbdet/Scripts/activate`

- run the run.py script:

`python hbdet/run.py`

The software will now run thorugh your dataset and gerate annotations for every (readable) soundifle within the dataset. While running, a spreadsheet, called stats.csv is continuously updated showing information on the annotations for every file (do not open while program is still running, because the program wont be able to access it).

The program will create a directory called `generated_annotatoins` in the project directory. It will then create a directory corresponding to the date and time that you started the annotation process. Within that directory you will find a directory `thresh_0.5` corresponding to all annotations with a threshold of 0.5. Furthermore you will find the `stats.csv` spreadsheet.

If you have chosen option 2 (or 0) you will also find a directory `thresh_0.x` where the x stands for the custom threshold you specified in the **simple_config.yml** file. Within the `thresh` directories you will find the name of your dataset. 

If you have chosen option 4 (or 0) you will find a directory `analysis` within the dataset directory. In that directory you will find spreadsheets for hourly presence and hourly counts, as well as folders containing visualizations of the hourly presence and hourly counts. The folder name starts with the date and time that they were computed.