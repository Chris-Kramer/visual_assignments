#!/usr/bin/env bash

echo "Building and activating environment"
#Environment name
VENVNAME=as4-cmk

#create venv
python3 -m venv $VENVNAME

#Activate environment
source $VENVNAME/bin/activate

echo "upgrading pip and installing dependencies"
#Upgrade pip
pip install --upgrade pip

# problems when installing from requirements.txt
test -f requirements.txt && pip install -r requirements.txt

#navigate to src folder
cd src

echo "running script"
#run script @$ means pass arguments from bash script to python script
python3 nn-mnist.py $@

echo "Deactivating and removing venv"
#deactivate environment
deactivate

#back to parent dir
cd ..

#remove venv
rm -rf $VENVNAME

echo "DONE!"