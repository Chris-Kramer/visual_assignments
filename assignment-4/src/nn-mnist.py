#!/usr/bin/env python
"""
---------- Import libs -----------
"""
import sys,os
sys.path.append(os.path.join(".."))

#Utility function - Neural networks with numpy
from utils.neuralnetwork import NeuralNetwork
import argparse
import numpy as np

# Machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
"""
----------- Main ------------
"""
def main():
    """
    ---------- Parameters ----------
    """
    #Create an argument parser from argparse
    ap = argparse.ArgumentParser(description = "[INFO] Classify MNIST data and print out performance report")
    
    #size of training data in percentage
    ap.add_argument("-trs", "--train_size",
                    required = False,
                    default = 0.8,
                    type = float,
                    help = "The size of the training data as a percentage. DEFAULT = 0.8 (80%)")
    
    #size of test data in percentage 
    ap.add_argument("-tes", "--test_size",
                    required = False,
                    default = 0.2,
                    type = float,
                    help = "The size of the test data as a percentage. DEFAULT = 0.2 (20%)")
    
    #number of epochs
    ap.add_argument("-ep", "--epochs",
                    required = False,
                    default = 500,
                    type = int,
                    help = "The number of epochs that should run. DEFAULT = 500")
    
    #Hidden layers
    ap.add_argument("-l", "--layers",
                    required = False,
                    default = [32, 16],
                    nargs = "*",
                    #action="append",
                    #type = list,
                    help = "Hidden layers as a list (max three items in the list). DEFAULT = 32 16)")
  
    #Create an argument parser from argparse
    args = vars(ap.parse_args())
    
    #Save in variables for readability
    epoch_n = args["epochs"]
    layers = args["layers"]
    trs = args["train_size"] #training size
    tes = args["test_size"] #test size
    
    """
    ---------- Get and transform data ----------
    """
    #Fetch data
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    
    #Convert to numpy arrays
    X = np.array(X) #data
    y = np.array(y) #labels
    
    #Rescale from between 0-255 to between 0-1
    X = (X - X.min())/(X.max() - X.min())
    
    #Create training data and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        train_size = trs,
                                                        test_size = tes)
    # convert labels from integers to vectors (binary)
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    
    """
    ----------- Train network -----------
    """
    
    layers_length = len(layers)
    
    #If there are three layers
    if (layers_length == 3):         
        #Train network
        print("[INFO] training network...")
        nn = NeuralNetwork([X_train.shape[1], int(layers[0]), int(layers[1]), int(layers[2]), 10])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs= epoch_n)
    
    #If there are two layers
    elif (layers_length == 2):
        #Train network
        print("[INFO] training network...")
        nn = NeuralNetwork([X_train.shape[1], int(layers[0]), int(layers[1]), 10])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs= epoch_n)
        
    #If there is one layers    
    elif (layers_length == 2):
        #Train network
        print("[INFO] training network...")
        nn = NeuralNetwork([X_train.shape[1], int(layers[0]), 10])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs= epoch_n)
    
    """
    ------------ Evaluate network------------
    """
    # Evaluate network
    print(["[INFO] evaluating network..."])
    predictions = nn.predict(X_test) # We take the model and predict the test class
    predictions = predictions.argmax(axis=1)
    print(classification_report(y_test.argmax(axis=1), predictions))
    
# Define behaviour when called from command line
if __name__ == "__main__":
    main()

