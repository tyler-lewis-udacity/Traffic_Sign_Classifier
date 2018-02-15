#!/bin/bash

# This script will activate a conda environment and launch a jupyter notebook:
# (Use "cpu"for laptop, "gpu" for desktop)

cd "/home/ty/Udacity/T1/P2_traffic_sign_classifier"
source "/home/ty/anaconda3/bin/activate" tensorflow-gpu
jupyter notebook Traffic_Sign_Classifier.ipynb

