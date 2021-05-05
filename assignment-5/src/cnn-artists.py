#!/usr/bin/env python3
"""
---------- Import libraries ----------
"""
#System tools
import os
import sys
sys.path.append(os.path.join("..")) #For importing homebrewed functions

#Argparse
import argparse
from argparse import RawTextHelpFormatter # Formatting -help

#Plotting
import matplotlib.pyplot as plt

#homebrewed functions
import utils.cnn_utils as cnn_utils

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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
    #Otherwise it will warn that some labels arent predicted (but this is natural with small data sets)
    #Moreover the report makes it clear if the model performs poorly
    warnings.filterwarnings('ignore')
    """
    -------------- Parameters -------------
    """
    #Create an argument parser from argparse
    ap = argparse.ArgumentParser(description = "[INFO] CNN for classifying painters",
                                formatter_class = RawTextHelpFormatter)
    
    #Use this flag if you need to split the data. It takes a folder with categories as input
    ap.add_argument("-sd", "--split_data",
                    required = False,
                    default = None,
                    type = str,
                    help =
                    "[INFO] Use this flag if you need to split the data in training and test data \n"
                    "[INFO] It takes a folder as input. The folder must be located in the 'data' folder \n"
                    "[INFO] The train-split divide will be 80/20%% \n"
                    "[TYPE] str \n"
                    "[DEFAULT] None \n"
                    "[EXAMPLE] --split_data shapes")
    
    # dir to training data
    ap.add_argument("-td", "--train_data",
                    required = False,
                    default = "small_training",
                    type = str,
                    help =
                    "[INFO] The folder with training data. Must be a subfolder in the 'data' folder \n"
                    "[TYPE] str \n"
                    "[DEFAULT] small_training \n"
                    "[EXAMPLE] --train_data training")
    
    # dir to test data
    ap.add_argument("-vd", "--validation_data",
                    required = False,
                    default = "small_validation",
                    type = str,
                    help =
                    "[INFO] The folder with validation data. Must be a subfolder in the 'data' folder \n"
                    "[TYPE] str \n"
                    "[DEFAULT] small_validation \n"
                    "[EXAMPLE] --validation_data validation")
    
    # Name for image of architecture
    ap.add_argument("-a", "--architecture_out",
                    required = False,
                    default = "model_architecture.png",
                    type = str,
                    help =
                    "[INFO] The name of the output image with model architecture \n"
                    "[TYPE] str \n"
                    "[DEFAULT] model_architecture.png \n"
                    "[EXAMPLE] --architecture_out test_architecture.png")
    
    #Name for plot with performance
    ap.add_argument("-p", "--performance_out",
                    required = False,
                    default = "performance.png",
                    type = str,
                    help =
                    "[INFO] The filename for output plot with performance \n"
                    "[TYPE] str \n"
                    "[DEFAULT] performance.png \n"
                    "[EXAMPLE] --performance_out performance.png")
    
    # image size
    ap.add_argument("-i", "--image_size",
                    required = False,
                    default = [120, 120],
                    nargs = "*",
                    type = int,
                    help =
                    "[INFO] The dimensions of resized pictures as a list of ints [height, width] \n"
                    "[TYPE] list of ints \n"
                    "[DEFAULT] 120 120 \n"
                    "[EXAMPLE] --image_size 60 60")
    
    # kernel size
    ap.add_argument("-k", "--kernel_size",
                    required = False,
                    default = [3, 5],
                    nargs = "*",
                    type = int,
                    help =
                    "[INFO] The size of two convolutionals kernels that are used in the first and second layer \n"
                    "[INFO] First value represents the kernel size of first conv2d layer \n"
                    "[INFO] Second value represents sencond conv2d layer \n"
                    "[TYPE] str \n"
                    "[DEFAULT] 3 5 \n"
                    "[EXAMPLE] --kernel_size 5 7")
    # filters
    ap.add_argument("-f", "--filters",
                    required = False,
                    default = [32, 50],
                    nargs = "*",
                    type = int,
                    help =
                    "[INFO] The amount of filters in the convolutional layers (there are two) \n"
                    "[INFO] Argument is a list of ints (length of two) \n"
                    "[INFO] First value is amount of filters in first conv2d layer \n"
                    "[INFO] Second value is amount of filters in second conv2d layer \n"
                    "[TYPE] list of ints \n"
                    "[DEFAULT] 32 50 \n"
                    "[EXAMPLE] --filters 32 64")
    
    # pool size
    ap.add_argument("-pz", "--pool_size",
                    required = False,
                    default = [2, 2],
                    nargs = "*",
                    type = int,
                    help = 
                    "[INFO] The pool size for pooling layers (there are two) as a list of ints \n"
                    "[INFO] First value represents first pooling layer, second value represents second pooling layer \n"
                    "[TYPE] list of ints \n"
                    "[DEFAULT] 2 2 (2x2 and 2x2) \n"
                    "[EXAMPLE] --pool_size 3 3")
    # strides
    ap.add_argument("-st", "--strides",
                    required = False,
                    default = [2, 2],
                    nargs = "*",
                    type = int,
                    help =
                    "[INFO] The strides for each pooling layer (there are two) \n"
                    "[INFO] First value represents strides ind first pooling layer \n"
                    "[INFO] Second value represents strides in second pooling layer \n"
                    "[TYPE] list of ints \n"
                    "[DEFAULT] 2 2 (2x2 and 2x2) \n"
                    "[EXAMPLE] --strides 3 3")
    
    # PAdding type
    ap.add_argument("-pa", "--padding",
                    required = False,
                    default = ["same", "same"],
                    nargs = "*",
                    type = str,
                    help =
                    "[INFO] The padding type for each convolutional layer (there are two) \n"
                    "[INFO] First value represents first pooling layer, second value represents second pooling layer \n"
                    "[TYPE] str \n"
                    "[DEFAULT] same same \n"
                    "[EXAMPLE] --padding valid valid")
    
    # Activation layers
    ap.add_argument("-al", "--activation_layers",
                    required = False,
                    default = ["relu", "relu", "relu", "softmax"],
                    nargs = "*",
                    type = str,
                    help =
                    "[INFO] Each activation level (There are four) \n"
                    "[INFO] I recommend not changing these layers \n"
                    "[INFO] However, if you are doing binary classification sigmoid might be perform than softmax \n"
                    "[TYPE] str \n"
                    "[DEFAULT] relu relu relu softmax \n"
                    "[EXAMPLE] --activation_layers relu relu relu sigmoid")
    # Learning rate
    ap.add_argument("-lr", "--learning_rate",
                    required = False,
                    default = 0.01,
                    type = float,
                    help =
                    "[INFO] The learning rate for stochastic gradient descent \n"
                    "[TYPE] float \n"
                    "[DEFAULT] 0.01 \n"
                    "[EXAMPLE] --learning_rate 0.001")
    # batch size
    ap.add_argument("-bs", "--batch_size",
                    required = False,
                    default = 32,
                    type = int,
                    help =
                    "[INFO] The batch size for processing \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 32 \n"
                    "[EXAMPLE] --batch_size 32")
    
    #number of epochs
    ap.add_argument("-ep", "--epochs",
                    required = False,
                    default = 20,
                    type = int,
                    help =
                    "[INFO] The number of epochs that should run \n"
                    "[TYPE] 20 \n"
                    "[DEFAULT] 20 \n"
                    "[EXAMPLE] --epochs 20")

    args = vars(ap.parse_args())
    
    #save arguments in variables (for readability)
    architecture_out = os.path.join("..", "output", args["architecture_out"])
    performance_out = os.path.join("..", "output", args["performance_out"])
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
    #If the data is already split between test and training data
    if args["split_data"] == None:
        
        #Save data paths in variables
        training_data = os.path.join("..", "data",  args["train_data"])
        validation_data = os.path.join("..", "data", args["validation_data"])
        
        #----- Find and create labels -----
        print("finding labels...")
        trainY = cnn_utils.get_y(training_data)
        testY = cnn_utils.get_y(validation_data)
        label_names = cnn_utils.get_label_names(validation_data)
    
        #----- Preprocess images -----
        print("Preprocessing images...")
        # Create training and test data
        trainX = cnn_utils.get_x(training_data, (image_size[0], image_size[1]))
        testX = cnn_utils.get_x(validation_data, (image_size[0],image_size[1]))
    
    #if the data needs to be split
    else:
        #Save data path in variable
        split_data = os.path.join("..", "data", args["split_data"])
        #Get X
        print("Preprocessing images...")
        X = cnn_utils.get_x(split_data, (image_size[0], image_size[1]))
        # Get y and label names
        print("finding labels...")
        y = cnn_utils.get_y(split_data)
        label_names = cnn_utils.get_label_names(split_data)
        
        #Create training data and test data
        trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.2)
        
    #----- Binarize labels ------
    print("binarizing labels...")
    #Binariser
    lb = LabelBinarizer()
    #Transform labels 0-9 into binary labels
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    
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
    model.add(Dense(len(label_names)))
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
    plot_model(model, to_file = architecture_out, show_shapes=True, show_layer_names=True)
    
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
    cnn_utils.plot_history(H, epochs, performance_out)
    
    #Print classification report
    print("evaluating model...")
    predictions = model.predict(testX, batch_size = batch_size)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_names))
    
if __name__ == "__main__":
    main()

