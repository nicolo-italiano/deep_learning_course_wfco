# Deep Learning Project - Group 74

This repository hosts the scripts developed for Deep Learning Course Project 2025/2026 - Group 74. The purpose of this project is to train a feed-forward neural network over Wind Farm Control (WFC) optimizations performed on multiple wind farm layouts, with the goal to accurately and quickly predict the optimal yaw angle of all wind turbines, for different wind conditions.

## Get started
To get started and download all the required packages, run the following command in a dedicated Python environment:
```
pip install -r requirements.txt
```
## Data
The preprocessed data needed to run the main notebook (*01_main_notebook.ipynb*) can be downloaded at this [link](https://dtudk-my.sharepoint.com/:u:/g/personal/nicit_dtu_dk/EX96_QZYCQZMhQgLV_bEnl0BgosY2sJLT1s48-sCzHR6Cw?e=jhTNmx), using a DTU account. Then, you can create a folder *data* and save it as *"5wt_dataset_1000_slsqp_simple_complete.pt"*.

## Structure

The repository loosely aligns with the [Cookiecutter for data science](https://cookiecutter-data-science.drivendata.org/) template. Specifically, it is structured as follow:

```text
deep_learning_course_wfco/
|   .gitignore                                   # .gitignore file
|   README.md                                    # README file
|
+---models
|       model_reduced_512_0.001_100_1000x1000_job0.pth
|       model_reduced_512_0.001_100_1000x1000x1000_job1.pth
|       model_reduced_512_0.0001_100_1000x1000x1000x1000_job2.pth
|       model_reduced_512_0.00075_100_1000x1000_job3.pth
|       model_reduced_512_0.00075_100_1000x1000x1000_job4.pth
|       model_reduced_512_0.00075_100_1000x1000x1000x1000_job5.pth
|       model_reduced_512_0.0005_100_1000x1000_job6.pth
|       model_reduced_512_0.0005_100_1000x1000x1000_job7.pth
|       model_reduced_512_0.0005_100_1000x1000x1000x1000_job8.pth
|
+---notebooks
|       00_data_preprocessing.ipynb              # notebook used to preprocess the data before training
|       01_main_notebook.ipynb                   # main notebook replicating the results of the project
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
