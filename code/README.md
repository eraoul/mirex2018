# seq2seqP4P:  MIREX 2018
### Eric Nichols (epnichols@gmail.com)

# Introduction
This folder contains a python program that is my entry into the 
2018 MIREX tasx "Patterns for Prediction". The program only works on 
for task 1 (sequence completion) for monophonic inputs.

This is a python program that requires keras/tensorflow for running
a pre-trained machine learning model. 

## Files:

* `__init__.py`: Python init file
* `data_processing.py`: Utility functions for input data
* `generate_continuation_mono.py`: The main python program
* `README.md`: This file
* `s2s_mono_continuation.h5`: A large HDF5 file contianing the model 
parameters

## Installation

To install the program, it is recommended to set up a python 
virtual environment such as conda. 

Create a virtual environment for **python 3.6**.

Inside the virtual environment, install:
* **tensorflow** 1.9  -- N.B. Install `tensorflow-gpu` if you have a GPU available 
for speeding up computation. GPU use also requires installing CUDA and 
cuDNN. `pip install tensorflow` or `pip install tensorflow-gpu`. See 
tensorflow.org for install notes.
* **keras** 2.2 -- `pip install keras`
* **pandas** -- `pip install pandas`
* **numpy** 1.14 -- Just verify installation. Should be installed 
by the tensorflow install process.

# Run

Run the command with the syntax:

`python3 generate_continuation_mono.py -i <INPUT_PATH> -o <OUTPUT_PATH>`