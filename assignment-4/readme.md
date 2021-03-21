# Assignment 4 - Classification benchmarks
**Christoffer Kramer**  
**21-03-2021**  
Classifier benchmarks using Logistic Regression and a Neural Network  
This assignment builds on the work we did in class and from session 6.  
You'll use your new knowledge and skills to create two command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal. These scripts can then be used to provide easy-to-understand benchmark scores for evaluating these models.  
You should create two Python scripts. One takes the full MNIST data set, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal. The other should take the full MNIST dataset, train a neural network classifier, and print the evaluation metrics to the terminal.  

## How to run  
**Step 1: Clone repo**  
- open terminal  
- Navigate to destination for repo  
- type the following command  
```console
 git clone https://github.com/Chris-Kramer/visual_assignments.git
```  
**step 2: Run bash script:**  
- Navigate to the folder "assignment-4".  
```console
cd assignment-4
```  
- Use the bash script _run-script.sh_ or _run_lr-mnist.sh_ to set up environment and run the scripts:  
```console
bash run_nn-mnist.sh
```  
```console
bash run_lr-mnist.sh
```
The bash scripts will print out a performance report when each script is run.

## Output
The output is printed directly to the terminal with both scripts.

## Parameters
Both scripts takes parameters, they have already ben supplied with default values, but feel free to change them.

### run_lr-mnist.sh
- train_size: The size of the training data as a percentage  
DEFAULT = 0.8  
- test_size: The size of the test data as a percentage  
DEFAULT = 0.2  
Example:  
```console
bash run_lr-mnist.sh --train_size 0.75 --test_size 0.25
```
### run_nn-mnist.sh
- train_size: The size of the training data as a percentage.  
DEFAULT = 0.8  
- test_size: The size of the test data as a percentage  
DEFAULT = 0.2  
- epochs: The number of epochs that should run
DEFAULT = 500  
- layers  
DEFAULT = 32 16

Example:  
```console
bash run_nn-mnist.sh --train_size 0.75 --test_size 0.25 --epochs 550 --layers 16 10 5
```