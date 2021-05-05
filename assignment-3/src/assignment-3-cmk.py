"""
---------- Import libs ----------
"""
import os
import sys
#sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import argparse
from argparse import RawTextHelpFormatter # Formatting -help
"""
---------- Main function ----------
"""
def main():
    
    """
    ---------- Parameters ----------
    """
    #Create an argument parser from argparse
    ap = argparse.ArgumentParser(description = "[INFO] Finding contours and cropping image",
                                 formatter_class = RawTextHelpFormatter)
    #Image 
    ap.add_argument("-im", "--image",
                    required = False,
                    default = "jefferson_memorial.jpg",
                    type = str,
                    help = 
                    "[INFO] File name of the image. Must be located in the folder 'data' \n"
                    "[TYPE] Str \n"
                    "[DEFAULT] jefferson_memorial.jpg \n"
                    "[EXAMPLE]--image test.jpg")
    
    #Pixels left from center (x-axis)
    ap.add_argument("-x_left", "--x_pixels_left",
                    required = False,
                    default = 750,
                    type = int,
                    help =
                    "[INFO] ROI in pixels left from center (integer) \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 750 \n"
                    "[EXAMPLE] --x_pixels_left 700")
    
    #Pixels right from center (x-axis)
    ap.add_argument("-x_right", "--x_pixels_right",
                    required = False,
                    default = 700,
                    type = int,
                    help = 
                    "[INFO] ROI in pixels right from center \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 700 \n"
                    "[EXAMPLE] --x_pixels_right 600")
    
    #Pixels up from center (y-axis)
    ap.add_argument("-y_up", "--y_pixels_up",
                    required = False,
                    default = 750,
                    type = int,
                    help = 
                    "[INFO] ROI in pixels up from center \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 750 \n"
                    "[EXAMPLE] --y_pixels_up 800")
    
    #Pixels down from center (y-axis)
    ap.add_argument("-y_down", "--y_pixels_down",
                    required = False,
                    default = 1175,
                    type = int,
                    help =
                    "[INFO] ROI in pixels down from center \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 1175 \n"
                    "[EXAMPLE] --y_pixels_down 1000")
    
    #Blur kernel
    ap.add_argument("-bk", "--blur_kernel",
                    required = False,
                    default = 5,
                    type = int,
                    help =
                    "[INFO] The size of the blur kernel (Gaussian blur) \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 5 \n"
                    "[EXAMPLE] --blur_kernel 5")
    
    #Lower threshold
    ap.add_argument("-lt", "--lower_thresh",
                    required = False,
                    default = 100,
                    type = int,
                    help =
                    "[INFO] The lower threshold for gaussian blur \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 100 \n"
                    "[EXAMPLE] --lower_thresh 150")
    
    #Upper threshold
    ap.add_argument("-ut", "--upper_thresh",
                    required = False,
                    default = 150,
                    type = int,
                    help =
                    "[INFO] The lower threshold for gaussian blur \n"
                    "[TYPE] int \n"
                    "[DEFAULT] 150 \n"
                    "[EXAMPLE] --upper_thresh 250")
    
    ap.add_argument("-roi", "--roi_output",
                    required = False,
                    default = "image_with_ROI.jpg",
                    type = str,
                    help = 
                    "[INFO] Filename for output image with ROI, will be located in the folder 'output' \n"
                    "[TYPE] str \n"
                    "[DEFAULT] image_with_ROI.jpg \n"
                    "[EXAMPLE] --roi_output test_img_ROI.jpg")
    
    ap.add_argument("-cp", "--cropped_output",
                    required = False,
                    default = "image_cropped.jpg",
                    type = str, 
                    help = 
                    "[INFO] Filename for cropped output image, will be located in the folder 'output' \n"
                    "[TYPE] str \n"
                    "[DEFAULT] image_cropped.jpg \n"
                    "[EXAMPLE] --cropped_output test_img_cropped.jpg")
    
    ap.add_argument("-co", "--contour_output",
                    required = False,
                    default = "image_contours.jpg",
                    type = str,
                    help =
                    "[INFO] Filename for image with contour lines, will be located in the folder 'output' \n"
                    "[TYPE] str \n"
                    "[DEFAULT] image_contours.jpg \n"
                    "[EXAMPLE] --contor_output test_img_contours.jpg")
    
    # Get values from arguments
    args = vars(ap.parse_args())
    
    """
    ------ Calculate values for cropping and Region of interest ------
    """
    print("Finding ROI ...")
    #Image
    img_path = os.path.join("..", "data", args["image"])
    image = cv2.imread(img_path) 

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
    print("Drawing ROI ..")
    #Draw rectangle
    img_with_ROI = cv2.rectangle(image.copy(),
                      start_point,
                      end_point,
                      (0,255,0),
                      2)
    #save image
    roi_out = os.path.join("..", "output", args["roi_output"])
    cv2.imwrite(roi_out, img_with_ROI)
    
    """
    ---------- Crop image ------------
    """
    print("cropping image ...")
    cropped_image = image[(centerY - y_pixels_up):(centerY + y_pixels_down),
                   (centerX - x_pixels_left):(centerX + x_pixels_right)]
    #Save image
    cropped_out = os.path.join("..", "output", args["cropped_output"])
    cv2.imwrite(cropped_out, cropped_image)
    
    """
    Blur and canny edge detection
    """
    print("Blurring and using canny edge detection ...")
    #Convert to greyscale
    grey_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    #Gaussian blur
    blurred = cv2.GaussianBlur(grey_cropped, (args["blur_kernel"], args["blur_kernel"]), 0)
    #Canny blur
    canny = cv2.Canny(blurred, args["lower_thresh"], args["upper_thresh"])

    """
    ----------- Find and draw contours -----------
    """
    print("Drawing contours ...")
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
    contour_output = os.path.join("..", "output", args["contour_output"])
    cv2.imwrite(contour_output, image_contours)
    
#Define behaviour when called from command line
if __name__ == "__main__":
    main()
