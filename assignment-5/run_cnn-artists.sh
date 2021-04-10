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

# navigate to data folder
#cd data

#echo "unzipping data"
# Unzip csv file (The file is to big to upload)
#unzip validation.zip
#unzip training.zip

# Move to source folder
#cd ../src
cd src

echo "running script"
# Run python script
python3 cnn-artists.py $@

# Move to data folder
#cd ../data

#echo "removing unzipped data"
# Remove unzipped folders (this is done, so I can push the repo without hitting the limit for data storage)
#rm -rf training 
#rm -rf validation

echo "deactivating and removing environment"
# Deavtivate environment
deactivate

# Move to home directory
cd ..

# Remove virtual environment (I'm just doing this to test if this is possible)
rm -rf $VENVNAME

#Print this to the screen 
echo "Done! The results can be found in the folder 'output'"