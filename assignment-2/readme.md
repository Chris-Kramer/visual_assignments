# Assignment 2 - Simple image search
**Christoffer Kramer**  
**18-02-2021**  

Creating a simple image search script  
Download the Oxford-17 flowers image data set, available at this link:https://www.robots.ox.ac.uk/~vgg/data/flowers/17/  

Choose one image in your data that you want to be the 'target image'. Write a Python script or Notebook which does the following:  
* Use the cv2.compareHist() function to compare the 3D color histogram for your target image to each of the other images in the corpus one-by-one.  

* In particular, use chi-square distance method, like we used in class. Round this number to 2 decimal places.  

* Save the results from this comparison as a single .csv file, showing the distance between your target image and each of the other images. The .csv file should show the filename for every image in your data except the target and the distance metric between that image and your target. Call your columns: filename, distance.  


## How to run

**Step 1: Clone repo**
- open terminal
- Navigate to destination for repo
- type the following command
    - _git clone https://github.com/Chris-Kramer/visual_assignments.git_  

**step 2: set op virtual enviroment:**
- Navigate to the folder "assignment-2".
    - _cd assignment-2_  
- Set up enviroment by running one of the following command:  
    - _source venv_assignment_2_cmk/bin/activate_  
        
**Step 3: Download requirements**
- _pip install -r requirements.txt_
        
**Step 4: Execute script**
- navigate to the folder with the script (src)
    - _cd src_
- run script
    - _python3 assignment-2-cmk.py_  
    
The script will after a couple of minutes start printning the filename and the distance between the image and target image. 

### Output
The output is a csv-file which can be found in the folder "output" under the name "results.csv".

### Parameters
The  parameters are the following:  

__Path to folder with images__  

folder_path = os.path.join("..", "data", "17flowers", "jpg")  

__Target image from the folder with images__  

target_img = cv2.imread(os.path.join("..", "data", "17flowers", "jpg", "image_0006.jpg"))  

__Destination and filename for output__  

output = os.path.join("..", "output", "results.csv")  


They are located in line 15-22. If you whish to change them. 