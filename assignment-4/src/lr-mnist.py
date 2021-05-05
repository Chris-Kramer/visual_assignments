#!/usr/bin/env python3
"""
---------- Import libs -----------
"""
#System tools
import os
import sys
sys.path.append(os.path.join(".."))

#Data
import numpy as np

# Import teaching utils
import utils.classifier_utils as clf_util

#Argparse
import argparse
from argparse import RawTextHelpFormatter # Formatting -help

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""
----------- Main ------------
"""
def main():
    """
    ---------- Parameters ----------
    """
    #Create an argument parser from argparse
    ap = argparse.ArgumentParser(description = "[INFO] Classify MNIST data and print out performance report",
                                formatter_class = RawTextHelpFormatter)
    
    #size of test data as a percentage float 
    ap.add_argument("-tes", "--test_size",
                    required = False,
                    default = 0.2,
                    type = float,
                    help =
                    "[INFO] The size of the test data as a float percentage. \n"
                    "[INFO] The training size will be adjusted automatically \n"
                    "[TYPE] float \n"
                    "[DEFAULT] 0.2 \n"
                    "[EXAMPLE] --test_size 0.1")
    
    
    #Create an argument parser from argparse
    args = vars(ap.parse_args())
    
    #Save in variables for readability
    #trs = args["train_size"] #training size
    tes = args["test_size"] #test size

    """
    ---------- Create training data and test data -------
    """
    print("Fetching MNIST data ...")
    #Fetch data
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    
    #Convert to numpy arrays
    X = np.array(X) #Data
    y = np.array(y) #Classes
    
    print("Preprocessing data ...")
    #Create training data and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=tes)
    
    #re-scaling the features from 0-255 to between 0 and 1
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0
    
    print("Creating classifier ...")
    #Create classifier
    clf = LogisticRegression(penalty='none',
                             tol=0.1, 
                             solver='saga',
                             multi_class='multinomial').fit(X_train_scaled, y_train) #Fit classifier to our test data
    
    print("Predicting test data ...")
    #Predict test data 
    y_pred = clf.predict(X_test_scaled)
    
    #Create a classification report by comparing test data with predictions
    cm = metrics.classification_report(y_test, y_pred)
    print(cm) #Print to terminal
    
# Define behaviour when called from command line
if __name__ == "__main__":
    main()