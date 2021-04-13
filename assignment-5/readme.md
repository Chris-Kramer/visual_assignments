# Assignment 5 - Classification benchmarks
**Christoffer Kramer**  
**10-04-2021**  
Multi-class classification of impressionist painters  
So far in class, we've been working with 'toy' datasets - handwriting, cats, dogs, and so on. However, this course is on the application of computer vision and deep learning to cultural data. This week, your assignment is to use what you've learned so far to build a classifier which can predict artists from paintings.  
You can find the data for the assignment here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data  
Using this data, you should build a deep learning model using convolutional neural networks which classify paintings by their respective artists. Why might we want to do this? Well, consider the scenario where we have found a new, never-before-seen painting which is claimed to be the artist Renoir. An accurate predictive model could be useful here for art historians and archivists!  
For this assignment, you can use the CNN code we looked at in class, such as the ShallowNet architecture or LeNet. You are also welcome to build your own model, if you dare - I recommend against doing this.  
Perhaps the most challenging aspect of this assignment will be to get all of the images into format that can be fed into the CNN model. All of the images are of different shapes and sizes, so the first task will be to resize the images to have them be a uniform (smaller) shape.  
You'll also need to think about how to get the images into an array for the model and how to extract 'labels' from filenames for use in the classification report

## How to run  
**Step 1: Clone repo**  
- open terminal  
- Navigate to destination for repo  
- type the following command  
```console
 git clone https://github.com/Chris-Kramer/visual_assignments.git
```  
**step 2: Run bash script:**  
- Navigate to the folder "assignment-5".  
```console
cd assignment-5
```  
- Use the bash script _run_cnn-artists.sh_ to set up environment and run the script:  
```console
bash run_cnn-artists.sh
```  
The bash script will print out a performance report and save a summary and a graph in the folder _output_.

**Note:** Because of limitations regarding data storage on git, I'm only using a very small slice of the data. So the performance is attrociosly bad (around 18%). Moreover, when resizing images you might encounter the following warning "Corrupt JPEG data: 10 extraneous bytes before marker 0xd9". Ignore this, this   

## Parameters
For the heck of it, I tried to add a lot of parameters to this script, so the model can be costumized. It has default values for all parameters. If you wan't to fiddle around with the model I suggest taking a look at the picture Original_model_architecture.png so you can see how many layers, and what types of layers there are.   

- train_data: The folder to training data. DEFAULT = data/small_training   
- validation_data: The folder to test data. DEFAULT = data/small_validation
- output: The folder for output data. DEFAULT = output
- image_size: The size of resized pictures. DEFAULT = 120 120
- kernel_size: The size of two convolutionals kernels that are used in the first and second layer. DEFAULT = 3 5 (This means 3x3 and 5x5)
- filters: The size of two output filters from the convolutional layers (there are two). DEFAULT = 32 50
- pool_size: The pool size for pooling layers (there are two). DEFAULT = 2 2 (2x2 and 2x2)
- strides: The strides for each pooling layer (there are two). DEFAULT = 2 2 (2x2 and 2x2)
- padding: The padding type for each convolutional layer (there are two). DEFAULT = same same
- activation_layers: Each activation level (There are four). DEFAULT = relu relu relu softmax
- learning_rate: The learning rate for stochastic gradient descent. DEFAULT = 0.01
- batch_size: The size of the batch processing. DEFAULT = 32
- epochs: The number of epochs that should run. DEFAULT = 20
Example (This won't work since unless you have the same path to the data folder). 
```console
bash run_cnn-artists.sh --epochs 21 --kernel_size 3 7 --train_data data/training --validation_data data/validation
```