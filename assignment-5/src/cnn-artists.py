#!/usr/bin/env python3
"""
---------- Import libraries ----------
"""
#Standard tools
import os
import sys
import cv2
from pathlib import Path
import re
import numpy as np
import argparse
sys.path.append(os.path.join("..")) #For importing homebrewed functions

#Plotting
import matplotlib.pyplot as plt

#homebrewed functions
from utils.cnn_utility import Cnn_utils

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.models import Sequential #Our Model
from tensorflow.keras.layers import (Conv2D, #Our layers
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model #Plotting 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

#Warnings
import warnings
"""
----------- Main function -----------
"""
def main():
    #Surpress warnings this this is usefull, when using a small data set
    #Otherwise it will warn that some labels arent predicted (but this is natural with small data sets
    #Moreover the report makes it clear if the model performs poorly
    warnings.filterwarnings('ignore')
    """
    -------------- Parameters -------------
    """
    #Create an argument parser from argparse
    ap = argparse.ArgumentParser(description = "[INFO] CNN for classifying painters")
    
    # dir to training data
    ap.add_argument("-t", "--train_data",
                    required = False,
                    default = "data/small_training",
                    type = str,
                    help = "The folder to training data. DEFAULT = data/small_training")
    
    # dir to test data
    ap.add_argument("-v", "--validation_data",
                    required = False,
                    default = "data/small_validation",
                    type = str,
                    help = "The folder to test data. DEFAULT = data/small_validation")
    
    # output dir
    ap.add_argument("-o", "--output",
                    required = False,
                    default = "output",
                    type = str,
                    help = "The folder for output data. DEFAULT = output")
    
    # image size
    ap.add_argument("-i", "--image_size",
                    required = False,
                    default = [120, 120],
                    nargs = "*",
                    type = int,
                    help = "The size of resized pictures. DEFAULT = 120 120")
    
    # kernel size
    ap.add_argument("-k", "--kernel_size",
                    required = False,
                    default = [3, 5],
                    nargs = "*",
                    type = int,
                    help = "The size of two convolutionals kernels that are used in the first and second layer. DEFAULT = 3 5 (3x3 and 5x5")
    # filters
    ap.add_argument("-f", "--filters",
                    required = False,
                    default = [32, 50],
                    nargs = "*",
                    type = int,
                    help = "The size of two output filters from the convolutional layers (there are two). DEFAULT = 32 50")
    
    # pool size
    ap.add_argument("-pz", "--pool_size",
                    required = False,
                    default = [2, 2],
                    nargs = "*",
                    type = int,
                    help = "The size of pool size for pooling layers (there are two). DEFAULT = 2 2 (2x2 and 2x2")
    # strides
    ap.add_argument("-p", "--strides",
                    required = False,
                    default = [2, 2],
                    nargs = "*",
                    type = int,
                    help = "The strides for each pooling layer (there are two). DEFAULT = 2 2 (2x2 and 2x2")
    
    # PAdding type
    ap.add_argument("-pt", "--padding",
                    required = False,
                    default = ["same", "same"],
                    nargs = "*",
                    type = str,
                    help = "The padding type for each convolutional layer (there are two). DEFAULT = same same")
    
    # Activation layers
    ap.add_argument("-al", "--activation_layers",
                    required = False,
                    default = ["relu", "relu", "relu", "softmax"],
                    nargs = "*",
                    type = str,
                    help = "Each activation level (There are four). DEFAULT = relu relu relu softmax")
    # Learning rate
    ap.add_argument("-lr", "--learning_rate",
                    required = False,
                    default = 0.01,
                    type = float,
                    help = "The learning rate for stochastic gradient descent. DEFAULT = 0.01")
    # batch size
    ap.add_argument("-bs", "--batch_size",
                    required = False,
                    default = 32,
                    type = int,
                    help = "The size of the batch processing. DEFAULT = 32")
    
    #number of epochs
    ap.add_argument("-ep", "--epochs",
                    required = False,
                    default = 20,
                    type = int,
                    help = "The number of epochs that should run. DEFAULT = 20")

    args = vars(ap.parse_args())
    
    #save arguments in variables (for readability)
    #I'm using os.norm.path so the paths works on different OS
    training_data = os.path.normpath("../" + args["train_data"]) 
    validation_data = os.path.normpath("../" + args["validation_data"])
    output = os.path.normpath("../" + args["output"])
    image_size = args["image_size"]
    kernel_size = args["kernel_size"]
    filters = args["filters"]
    pool_size = args["pool_size"]
    strides = args["strides"]
    padding = args["padding"]
    activation_layer = args["activation_layers"]
    learning_rate = args["learning_rate"]
    batch_size = args["batch_size"]
    epochs = args["epochs"]
    
    """
    -------------- Preprocessing data ----------
    """
    #Import class with utility functions
    cnn_utils = Cnn_utils()
    
    #----- Find and create labels -----
    print("finding labels...")
    trainY = cnn_utils.get_y(training_data)
    testY = cnn_utils.get_y(validation_data)
    label_names = cnn_utils.get_label_names(validation_data)
    
    #----- Binarize labels ------
    print("binarizing labels...")
    #Binariser
    lb = LabelBinarizer()
    #Transform labels 0-9 into binary labels
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    
    #----- Preprocess images -----
    print("Preprocessing images...")
    # Create training and test data
    trainX = cnn_utils.get_x(training_data, (image_size[0], image_size[1]))
    testX = cnn_utils.get_x(validation_data, (image_size[0],image_size[1]))
    
    """
    ----------- Create model ----------
    """
    print("creating model...")
    # define model
    model = Sequential()

    # ------- first set of CONV => RELU => POOL -------
    #Conv
    model.add(Conv2D(filters[0], (kernel_size[0], kernel_size[0], ), 
                     padding=padding[0], 
                     input_shape= (image_size[0], image_size[1], 3)))
    #Relu
    model.add(Activation(activation_layer[0]))
    #Pool
    model.add(MaxPooling2D(pool_size=(pool_size[0], pool_size[0]), 
                           strides=(strides[0], strides[0])))

    # ------- second set of CONV => RELU => POOL ------
    #Conv
    model.add(Conv2D(filters[1], (kernel_size[1], kernel_size[1]), 
                     padding=padding[1]))
    #Relu
    model.add(Activation(activation_layer[1]))
    #Pool
    model.add(MaxPooling2D(pool_size=(pool_size[1], pool_size[1]), 
                           strides=(strides[1], strides[1])))

    # ------- FC => RELU --------
    #flatten
    model.add(Flatten())
    #Add layer
    model.add(Dense(500))
    #Relu
    model.add(Activation(activation_layer[2]))

    # -------- softmax classifier layer-------
    model.add(Dense(10))
    #Activation
    model.add(Activation(activation_layer[3]))
    
    # ------- Optimizer -------
    #Stochastic descent with learning rate 0.01
    opt = SGD(lr=learning_rate)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    
    # ------- Summarise model --------
    print("creating model summary...")
    #Create a summary of the model
    model.summary()
    #plot model
    plot_model(model, to_file = os.path.normpath(output + "/model_architecture.png"), show_shapes=True, show_layer_names=True)
    
    """
    ----------- Train and evaluate model --------------
    """
    print("training model...")
    # train model
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size= batch_size,
                  epochs= epochs,
                  verbose=1)
    
    #Plot performance pr. epoch
    cnn_utils.plot_history(H, epochs, os.path.normpath(output + "/performance.png"))
    
    #Create a summary of the model
    #Print classification report
    print("evaluating model...")
    predictions = model.predict(testX, batch_size = batch_size)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_names))
    
if __name__ == "__main__":
    main()

