#!/usr/bin/env bash

echo "Creating environment"
#Environment name
VENVNAME=as4-cmk

python3 -m venv $VENVNAME

#Activate environment
source $VENVNAME/bin/activate

echo "Upgrading pip and installing dependencies"
#Upgrade pip
pip install --upgrade pip

# problems when installing from requirements.txt
test -f requirements.txt && pip install -r requirements.txt

#navigate to src folder
cd src

echo "Running script"
#run script @$ means pass arguments from bash script to python script
python3 lr-mnist.py $@

echo "Deactivating and removing environment"
#deactivate environment
deactivate

#move to parent dir
cd ..

#Remove venv
rm -rf $VENVNAME

echo "DONE!"
