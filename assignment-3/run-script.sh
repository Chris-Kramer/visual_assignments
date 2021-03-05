#!/usr/bin/env bash

#Environment name
VENVNAME=assignment3_cmk

#Activate environment
source $VENVNAME/bin/activate

#Upgrade pip
pip install --upgrade pip

# problems when installing from requirements.txt
test -f requirements.txt && pip install requirements.txt

#Navigate to folder with script
cd src

#run script
python3 assignment-3-cmk.py

#deactivate environment
deactivate

echo "DONE! THE CROPPED PICTURES AND THE PICTURE OF CONTOUR LINES ARE LOCATED IN THE FOLDER'output'"