#!/usr/bin/env python
"""
----------Import libs---------
"""
import os
import sys
import csv
from pathlib import Path
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
"""
----------Parameters-----------
"""
#Path to folder with images
folder_path = os.path.join("..", "data", "17flowers", "jpg")

#Target image from the folder with images
target_img = cv2.imread(os.path.join("..", "data", "17flowers", "jpg", "image_0006.jpg"))

#Destination and filename for output
output = os.path.join("..", "output", "results.csv")
"""
----------Main function----------
"""

def main():
    
    #Create histogram
    target_hist = cv2.calcHist([target_img], [0,1,2], None, [8, 8, 8], [0,256, 0, 256, 0, 256])
    
    #normalize histogram
    target_hist = cv2.normalize(target_hist, target_hist, 0, 255, cv2.NORM_MINMAX)
    
    #Create csv
    with open(output, mode = "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "distance"])
        writer.writeheader() 
    
        #For each image in the folder with images
        for img in Path(folder_path).glob("*.jpg"):
        
            #save path as a string 
            img_name = str(img)
        
            #skip over the image, if it is the target image
            if img_name == str(target_img):
                continue
            
            else:
                #Read image
                img = cv2.imread(img_name)
    
                #calculate image histogram
                hist_img = cv2.calcHist([img], [0,1,2], None, [8, 8, 8], [0,256, 0, 256, 0, 256])
    
                #Normalize image histogram
                hist_img = cv2.normalize(hist_img, hist_img, 0, 255, cv2.NORM_MINMAX)
    
                #Compare target histogram with current image histogram
                distance = round(cv2.compareHist(target_hist, hist_img, cv2.HISTCMP_CHISQR), 2)
                
                #write csv row and print the results. 
                writer.writerow({"filename": img_name, "distance": distance})
                print(f"filename: {str(img_name)}, distance: {distance}")

#Define behaviour when called from commandline
if __name__=="__main__":
    main()