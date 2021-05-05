#!/usr/bin/env bash

VENVNAME=as5-cmk
echo "Creating environment"
python3 -m venv as5-cmk

echo "Activating environment"
source $VENVNAME/bin/activate

echo "Upgrading pip"
pip install --upgrade pip

echo "installing requirements"
# test for problems when installing from requirements.txt and install
test -f requirements.txt && pip install -r requirements.txt

# Move to source folder
cd src

echo "running script"
# Run python script
python3 cnn-artists.py $@

echo "deactivating and removing environment"
# Deavtivate environment
deactivate

# Move to home directory
cd ..

# Remove virtual environment
rm -rf $VENVNAME

#Print this to the screen 
echo "Done! The results can be found in the folder 'output'"