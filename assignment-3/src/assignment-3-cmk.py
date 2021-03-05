"""
---------- Import libs ----------
"""
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
from PIL import Image

"""
---------- Parameters ----------
"""
#Image
jeff_img = cv2.imread("../data/jefferson_memorial.jpg") 

#Center
(centerX, centerY) = jeff_img.shape[1] // 2, jeff_img.shape[0] // 2
    
#Pixels left and right from center
x_pixels_left = 750
x_pixels_right = 700
    
#Pixels up and down from center
y_pixels_up = 750
y_pixels_down = 1175
    
#Start and end point for rectangle
start_point = ((centerX - x_pixels_left), (centerY - y_pixels_up))
end_point = ((centerX + x_pixels_right), (centerY + y_pixels_down))

"""
---------- Main function ----------
"""
def main():
    """
    Create line with Region of Interest
    """
    
    #Draw rectangle
    img_with_ROI = cv2.rectangle(jeff_img.copy(),
                      start_point,
                      end_point,
                      (0,255,0),
                      2)
    #save image
    cv2.imwrite("../output/image_with_ROI.jpg", img_with_ROI)
    
    """
    Crop image
    """
    cropped_image = jeff_img[(centerY - y_pixels_up):(centerY + y_pixels_down),
                   (centerX - x_pixels_left):(centerX + x_pixels_right)]
    #Save image
    cv2.imwrite("../output/image_cropped.jpg", cropped_image)
    
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
    Find and draw contours
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
    cv2.imwrite("../output/image_letters.jpg", image_contours)
    
#Define behaviour when called from command line
if __name__ == "__main__":
    main()
