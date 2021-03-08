"""
---------- Import libs ----------
"""
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import argparse

"""
---------- Main function ----------
"""
def main():
    
    """
    ---------- Parameters ----------
    """
    ap = argparse.ArgumentParser(description = "[INFO] Finding contours and cropping image") #Create an argument parser from argparse
    #Image 
    ap.add_argument("-im", "--image", required = False,
                    type = str, help = "image e.g. 'data/img/image.jpg'")
    #Pixels left from center (x-axis)
    ap.add_argument("-x_left", "--x_pixels_left", required = False,
                    type = int, help = "pixels left from center (integer)")
    #Pixels right from center (x-axis)
    ap.add_argument("-x_right", "--x_pixels_right", required = False, 
                    type = int, help = "pixels right from center (integer)")
    #Pixels up from center (y-axis)
    ap.add_argument("-y_up", "--y_pixels_up", required = False,
                    type = int, help = "pixels up from center (integer)")
    #Pixels down from center (y-axis)
    ap.add_argument("-y_down", "--y_pixels_down", required = False,
                    type = int, help = "pixels up from center (integer)")
    args = vars(ap.parse_args())
    
    """
    ------ Calculate values for cropping and Region of interest ------
    """
    #Image
    image = cv2.imread(args["image"]) 

    #Center
    (centerX, centerY) = image.shape[1] // 2, image.shape[0] // 2
    
    #Pixels left and right from center
    x_pixels_left = args["x_pixels_left"]
    x_pixels_right = args["x_pixels_right"]
    
    #Pixels up and down from center
    y_pixels_up = args["y_pixels_up"]
    y_pixels_down = args["y_pixels_down"]
    
    #Start and end point for rectangle
    start_point = ((centerX - x_pixels_left), (centerY - y_pixels_up))
    end_point = ((centerX + x_pixels_right), (centerY + y_pixels_down))
    
    """
    ----- Create line with Region of Interest ------
    """
    
    #Draw rectangle
    img_with_ROI = cv2.rectangle(image.copy(),
                      start_point,
                      end_point,
                      (0,255,0),
                      2)
    #save image
    cv2.imwrite("output/image_with_ROI.jpg", img_with_ROI)
    
    """
    ---------- Crop image ------------
    """
    cropped_image = image[(centerY - y_pixels_up):(centerY + y_pixels_down),
                   (centerX - x_pixels_left):(centerX + x_pixels_right)]
    #Save image
    cv2.imwrite("output/image_cropped.jpg", cropped_image)
    
    """
    Blur and canny edge detection
    """
    #Convert to greyscale
    grey_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    #Gaussian blur
    blurred = cv2.GaussianBlur(grey_cropped, (5,5), 0)
    #Canny blur
    canny = cv2.Canny(blurred, 100, 150)

    """
    ----------- Find and draw contours -----------
    """
    #Find contours
    (contours, _) = cv2.findContours(canny.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
    #Draw contours
    image_contours = cv2.drawContours(cropped_image.copy(), #Draw contours on original
                        contours, #our list of contours
                         -1, #Which contour to draw (-1 draws all contours)
                        (0,255,0), #Contour color
                         2)
    #Save image
    cv2.imwrite("output/image_letters.jpg", image_contours)
    
#Define behaviour when called from command line
if __name__ == "__main__":
    main()
