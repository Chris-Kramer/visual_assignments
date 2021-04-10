#!/usr/bin/env python3
"""
------------- TO DO LIST (if time)-----------
* Create argparse parameters (This will enable me to create a cnn pipeline that can easily be used on different types of image data)
* Move the functions into a utility folder that can be imported (this will shorten the script)
* Make the functions into classes (This is mainly for training)
"""

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

#Plotting
import matplotlib.pyplot as plt

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.datasets import cifar10 #Our data
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
---------- Create functions ------------
"""
#Function for resizing images and converting it to value between zero and 1
def resize_imgs(folder_path, dimensions):
    #List of image sizes
    images = []
    #Folder  each folder in training data
    for folder in Path(folder_path).glob("*"):
        # For each file in the folder
        for image in Path(folder).glob("*"):
            #read image
            image = cv2.imread(str(image))
            #resize image
            image = cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)
            #append image to array and convert it to a value between 0 and 1
            images.append(image.astype("float")/255.)
    return images

#Function for plotting the models performance
def plot_history(H, epochs):
    # visualize performance
    plt.style.use("fivethirtyeight")
    fig = plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig("../output/performance.png")
    
def main():
    #Surpress warnings this this is usefull, when using a small data set
    #Otherwise it will warn that some labels arent predicted (but this is natural with small data sets
    #Moreover the report makes it clear if the model performs poorly
    warnings.filterwarnings('ignore')
    """
    -------------- Parameters -------------
    I don't have any argparse parameters yet, however if I find Time, I will set some in
    * Epochs
    * Size of images
    * Folder to training data
    * Folder to test data
    * Batch size
    * Even stuff like activation functions, padding and strides can be put to parameters.
    The stuff above will give me a working data cleaning and analysis pipeline, that can be used on lots of different data
    """
    
    """
    -------------- Preprocessing data ----------
    """
    print("finding labels...")
    #----- Find and create labels -----
    #Path to training folder with painters
    training_dir = os.path.join("..", "data", "small_training")
    #Names as a string
    label_names = []
    #Training labels
    trainY = []
    #Counter variable
    i = 0
    #For each folder in the training data
    for folder in Path(training_dir).glob("*"):
        #Find painters name
        painter = re.findall(r"(?!.*/).+", str(folder))
        #Append the painters names to the list "label_names"
        label_names.append(painter[0])
        #For each image in the folder
        for img in folder.glob("*"):
            #append the folder index (e.g. the label)
            trainY.append(i)
        #Increase counter by 1    
        i +=1 
        
     #Path to test folder with painters
    test_dir = os.path.join("..", "data", "small_validation")  
    #Test labels
    testY = []  
    #Counter
    i = 0
    #For each folder in the test data
    for folder in Path(test_dir).glob("*"):
        #For each image in the folder
        for img in folder.glob("*"):
            #Aooend the folder index (e.g. the label)
            testY.append(i)
        #Increase by 1
        i +=1 
    
    print("binarizing labels...")
    #----- Binarize labels ------
    #Binariser
    lb = LabelBinarizer()
    #Transform labels 0-9 into binary labels
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    
    print("resizing images...")
    #----- Resize images -----
    # paths
    filepath_training = os.path.join("..", "data", "small_training")
    filepath_validation = os.path.join("..", "data", "small_validation")
    
    # Create training and test data
    trainX = resize_imgs(filepath_training, (120, 120))
    testX =resize_imgs(filepath_validation, (120,120))
    
    #Convert data to numpy arrays
    testX = np.array(testX)
    trainX = np.array(trainX)
    
    """
    ----------- Create model ----------
    """
    print("creating model...")
    # define model
    model = Sequential()

    # ------- first set of CONV => RELU => POOL -------
    #Conv
    model.add(Conv2D(32, (3, 3), 
                     padding="same", 
                     input_shape= (120, 120, 3)))
    #Relu
    model.add(Activation("relu"))
    #Pool
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    # ------- second set of CONV => RELU => POOL ------
    #Conv
    model.add(Conv2D(50, (5, 5), 
                     padding="same"))
    #Relu
    model.add(Activation("relu"))
    #Pool
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    # ------- FC => RELU --------
    #flatten
    model.add(Flatten())
    #Add layer
    model.add(Dense(500))
    #Relu
    model.add(Activation("relu"))

    # -------- softmax classifier layer-------
    model.add(Dense(10))
    #Activation
    model.add(Activation("softmax"))
    
    # ------- Optimizer -------
    #Stochastic descent with learning rate 0.1
    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    
    # ------- Summarise model --------
    print("creating model summary...")
    #Create a summary of the model
    model.summary()
    #plot model
    plot_model(model, to_file = "../output/model_architecture.png", show_shapes=True, show_layer_names=True)
    
    """
    ----------- Train and evaluate model --------------
    """
    print("training model...")
    # train model
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size=32,
                  epochs=20,
                  verbose=1)
    #Plot performance pr. epoch
    plot_history(H,20)
    
    #Create a summary of the model
    #Print classification report
    print("evaluating model...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_names))
    
if __name__ == "__main__":
    main()

