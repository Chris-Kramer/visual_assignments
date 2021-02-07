# Assignment 1 - Basic image processing

Create or find small dataset of images, using an online data source such as Kaggle. At the very least, your dataset should contain no fewer than 10 images.

Write a Python script which does the following:

- For each image, find the width, height, and number of channels
- For each image, split image into four equal-sized quadrants (i.e. top-left, top-right, bottom-left, bottom-right)
- Save each of the split images in JPG format
- Create and save a file containing the filename, width, height for all of the new images.


## Folders and contents
- utils: Utility functions.
- data.zip: The raw data in a zip-file.
- data: data used in the assignment
    - new_imgs.csv: An auto-generated csv-file which list each new picture, its height, its widthm amount of channels and its file-name.
    - raw-img: Images used in the assignment. Further divivided into subfoldes.Each subfolder contains the folder "splitted_images". Which is where the splitted images are located.
        - cane
        - cavallo
        - elefante
        - farfalla
        - gallina
        - gatto
        - mucca
        - pecora
        - ragno
        - scoiatollo


## Metadata
Description: This dataset is a subset of a dataset from kaggle.com containing 26K pictures of animals in 10 categories. I have only used 10 pictures in each category. I have further added the folder "splitted_images" to each category.  
- Source: https://www.kaggle.com/alessiocorrado99/animals10
- For further details and more metadata please see this link: https://www.kaggle.com/alessiocorrado99/animals10/metadata