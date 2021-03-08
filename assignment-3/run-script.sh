#!/usr/bin/env bash

#Environment name
VENVNAME=assignment3_cmk

#Activate environment
source $VENVNAME/bin/activate

#Upgrade pip
pip install --upgrade pip

# problems when installing from requirements.txt
test -f requirements.txt && pip install requirements.txt

#parameters
image=${1:-"data/jefferson_memorial.jpg"}
x_pixels_left=${2:-750}
x_pixels_right=${3:-700}
y_pixels_up=${4:-750}
y_pixels_down=${5:-1175}

#run script
python3 assignment-3-cmk.py --image $image --x_pixels_left $x_pixels_left --x_pixels_right $x_pixels_right --y_pixels_up $y_pixels_up --y_pixels_down $y_pixels_down

#deactivate environment
deactivate

#Print to terminal
echo "DONE! THE CROPPED PICTURES AND THE PICTURE OF CONTOUR LINES ARE LOCATED IN THE FOLDER'output'"