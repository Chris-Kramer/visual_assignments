#!/usr/bin/env python
"""
---------- Import libs -----------
"""
import os
import sys
sys.path.append(os.path.join(".."))

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""
----------- Main ------------
"""
def main():
    """
    ---------- Parameters ----------
    """
    
    # Fetch data
    
    
# Define behaviour when called from command line
if name == "__main__":
    main()