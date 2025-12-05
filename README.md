# Deep Learning Project - Group 74

This repository hosts the scripts developed for Deep Learning Course Project 2025/2026 - Group 74. the purpose of this project is to train a feed-forward neural network over Wind Farm Control (WFC) optimizations performed on multiple wind farm layouts, with the goal to accurately and quickly predict the optimal yaw angle of all wind turbines, for different wind conditions.

## Get started
To get started and download all the required packages, run the following command in a dedicated Python environment:
```
pip install -r requirements.txt
```
## Data
To download the data needed to run the notebook...

## Structure

The repository loosely aligns with the [Cookiecutter for data science](https://cookiecutter-data-science.drivendata.org/) template. Specifically, it is structured as follow:

```text
deep_learning_course_wfco/
|   .gitignore                                   # .gitignore file
|   README.md                                    # README file
|
+---models
|       yaw_regression_model_2048_0.01_small.pth
|
+---notebooks
|       00_data_preprocessing.ipynb              # notebook used to preprocess the data before training
|
+---scripts
|       training.py                              # script used for training
|
+---slurm_scripts
|       nn_training.sh                           # SLURM script used to run the training script on DTU's HPC cluster Sophia
|
\---src
        utils.py                                 # source file containing functions used in the notebooks and in the scripts 
        __init__.py
```

*Nicolò Italiano*
