# Bayes by Backprop for Hydrologic Discharge Prediction

## What is this?
The repository contains the software and model code for *Bayes by Backprop for Hydrologic Discharge Prediction*. It
features all relevant source code for running the *Bayes-by-Backprop*-algorithm with the CAMELS-US-dataset as an
application. It is possible to train and evaluate models following all three architecture presented in the paper
(many-to-one, many-to-one with discharge and many-to-many). Moreover, it is possible to tweak the proposed
hyperparameters in order to experiment with different architectures.

## How to use the model code
The source code is written for Python 3.8, therefore we recommend setting up a virtual environment with this version of
Python. The source code depends on a couple of requirements which are listed in the *requirements.txt*-file.

A file names *camels_run.py* in der folder *camels_us* contains the necessary source code to train/evaluate a model.
The file contains a number of changable variables which are described in the following:

- *CAMELS_ROOT*: points to a specific subfolder of the CAMELS-US-dataset that is used for this project. Should not be changed unless you wish to use an own version of the dataset.
- *DEVICE*: the device used for the computations. Should normally not be changed.
- *PLOT_FREQUENCY*: the frequency to evaluate the model, plot the results of the evaluation and save an instance of the trained model. For example: 10 means that the model is evaluated every 10 epochs.
- *SAVE_MODEL*: should be set to *True* if instances of the trained model should be saved. *False* otherwise.
- *ONLY_EVAL*: should be set to *True* if training has already been done and one only wishes to evaluate the model. Note that in this case a valid path for loading the model has to be given as described below.
- *SAVE_PATH*: path where instances of the model should be saved (if the option is enabled).
- *LOAD_PATH*: path of the model instance that should be loaded in case *ONLY_EVAL* is enabled.

Lines 70-74 can be modified to change the used architecture or hyperparameters.

Note that it is not necessary to download separate datasets as the repository already contains all necessary data from
CAMELS-US. 

