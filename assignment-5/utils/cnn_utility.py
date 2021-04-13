#!/usr/bin/env python
import os
import sys
import cv2
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

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
def plot_history(H, epochs, output):
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
    fig.savefig(output)
        
if __name__=="__main__":
    pass