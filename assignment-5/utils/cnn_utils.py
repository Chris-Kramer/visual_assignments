#!/usr/bin/env python3
import os
import sys
import cv2
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

#Get X data and preprocess it so it has the correct dimensions. 
def get_x(folder_path, dimensions):
    #X data
    X = []
    #For each folder in training data
    for folder in Path(folder_path).glob("*"):
        #For each file in the folder
        for image in Path(folder).glob("*"):
            #read image
            image = cv2.imread(str(image))
            #resize image
            image = cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)
            #append image to array and convert it to a value between 0 and 1
            X.append(image.astype("float")/255.)
    #Convert X to numpy array
    X = np.array(X)
    return X
        
# Get the name of each folder (the labels as a string)
# folder_path needs to be a folder with subfolders
def get_label_names(folder_path):
    #Array for names
    label_names = []
    #For each folder in the training data
    for folder in Path(folder_path).glob("*"):
        #Find folders name
        label = re.findall(r"(?!.*/).+", str(folder))
        #Append the folders names to the list "label_names" (findall returns a list)
        label_names.append(label[0])
    return label_names
    
#Get y data (the label for each picture)
def get_y(folder_path):
    y = [] # A list of all images labels
    i = 0 #Counter
    #For each folder in the test data
    for folder in Path(folder_path).glob("*"):
        #For each image in the folder
        for img in folder.glob("*"):
            #Append the folder index (e.g. the label) to y
            y.append(i)
        i += 1
    #Convert Y to numpy array
    y = np.array(y)
    return y
    
#Function for plotting the models performance
def plot_history(H, epochs, output = "performance.png"):
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
    
# Create a subset of data (usefull when needing to create a small data set)
def subset_data(folder_path, destination, n_files):
    #folder which contains the sub directories
    #list sub directories 
    for root, dirs, files in os.walk(folder_path):
        #iterate through them
        for i in dirs: 
            #create a new folder with the name of the iterated sub dir
            path = destination + "%s/" % i
            os.makedirs(path)
            
            #take random sample, here 3 files per sub dir
            filenames = random.sample(os.listdir(folder_path + "%s/" % i ), n_files)
            
            #copy the files to the new destination
            for j in filenames:
                shutil.copy2(folder_path + "%s/" % i  + j, path)
                
# Find smallest image (usefull for getting an idea of, which input size to chose for neural network
def find_smallest_img(filepath):
    #Array for training data
    images =[]
    #List of image sizes
    image_sizes = []
    #Folder  each folder in training data
    for folder in Path(filepath).glob("*"):
        # For each file in the folder
        for image in Path(folder).glob("*"):
            #read image
            image = cv2.imread(str(image))
            #append image
            images.append(image)
            #Append the sum of the dimensions
            image_sizes.append(sum(image.shape))
    #Find the image
    smallest_image = images[image_sizes.index(min(image_sizes))]
    return smallest_image

# Find largest image (usefull for getting an idea of, which input size to chose for neural network
def find_largest_img(filepath):
    #Array for training data
    images =[]
    #List of image sizes
    image_sizes = []
    #Folder  each folder in training data
    for folder in Path(filepath).glob("*"):
        # For each file in the folder
        for image in Path(folder).glob("*"):
            #read image
            image = cv2.imread(str(image))
            #append image
            images.append(image)
            #Append the sum of the dimensions
            image_sizes.append(sum(image.shape))
    #Find the image
    largest_image = images[image_sizes.index(max(image_sizes))]
    return largest_image

if __name__=="__main__":
    pass #Don't run from terminal