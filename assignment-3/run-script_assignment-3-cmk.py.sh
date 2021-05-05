#!/usr/bin/env bash

echo "Building environment"
#Environment name
VENVNAME=assignment3_cmk

#create env
python3 -m venv $VENVNAME 

#Activate environment
source $VENVNAME/bin/activate

echo "Installing pip and requirements"
#Upgrade pip
pip install --upgrade pip

# problems and install requirements.txt
test -f requirements.txt && pip install -r requirements.txt

cd src 

echo "Running script"
#run script
python3 assignment-3-cmk.py $@

echo "Deactivating and removing environment"
#deactivate environment
deactivate

#remove environment
cd ..

rm -rf $VENVNAME

#Print to terminal
echo "DONE! THE CROPPED PICTURES AND THE PICTURE OF CONTOUR LINES ARE LOCATED IN THE FOLDER'output'"