# Assignment 3 - Edge detection
**Christoffer Kramer**  
**05-03-2021**  


The purpose of this assignment is to use computer vision to extract specific features from images. In particular, we're going to see if we can find text. We are not interested in finding whole words right now; we'll look at how to find whole words in a coming class. For now, we only want to find language-like objects, such as letters and punctuation.  
Download and save the image at the link below:  

https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG

Using the skills you have learned up to now, do the following tasks:  

- Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. Save this as image_with_ROI.jpg.  

- Crop the original image to create a new image containing only the ROI in the rectangle. Save this as image_cropped.jpg.  

- Using this cropped image, use Canny edge detection to 'find' every letter in the image  

- Draw a green contour around each letter in the cropped image. Save this as image_letters.jpg  


## How to run

**Step 1: Clone repo**
- open terminal
- Navigate to destination for repo
- type the following command
 ```console
 git clone https://github.com/Chris-Kramer/visual_assignments.git
 ```
**step 2: Run bash script:**
- Navigate to the folder "assignment-3".
```console
cd assignment-3
```  
- Use the bash script _run-script.sh_ to set up environment and run the script:  
```console
bash run-script.sh
```  
The bash script will print out `DONE! THE CROPPED PICTURES AND THE PICTURE OF CONTOUR LINES ARE LOCATED IN THE FOLDER'output'` when it is done running. 

## Output
The output is three images which can be found in the folder "output".

## Parameters
This script takes the following parameters in the listed order, it have already been supplied with default values. But you are welcome to try and change the default parameters. I have added a photo called "test.jpg" under the data folder, you can use it if you want to try another picture.

`--image` This is the path and filename of the image. 
Default = ../data/jefferson_memorial.jpg
`--x_pixels_left` How many pixels to the left of center should the region of interest include?
Default = 750
`--x_pixels_right`How many pixels to the right of center should the region of interest include?
Default = 700
`--y_pixels_up`How many pixels above the center should the region of interest include?
Default = 750
`--y_pixels_down`How many pixels below the center should the region of interest include?
Default = 1175

Example:
```console
bash run-script.sh data/test.jpg 100 100 100 100
```