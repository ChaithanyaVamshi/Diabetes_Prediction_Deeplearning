# Diabetes Prediction Using Deep Learning with Python

The Ultimate Guide for building Diabetes Classifier and Predicting Possibility of having Diabetes in a Patient by using Neural Networks with GridSearchCV.

![image](https://user-images.githubusercontent.com/31254745/156607115-c81cec52-7aa6-4f34-b730-bd1f4fd748a5.png)

## Introduction

Diabetes is a disease that occurs when the blood glucose is too high. Blood glucose is the main source of energy and comes from the food we eat. 
Over time, having too much glucose in the blood can cause health problems and can lead to death. Preventing Diabetes is important and with advancements in AI, predicting the possibility of having Diabetes allows a person to take steps to manage diabetes and treat such situations at early stages making sure that more people can live healthy lives.

In this project, I will demonstrate and show how we can harness the power of Deep Learning and apply it in healthcare. I will walk you through the entire process of how to classify and predict the possibility of having diabetes in a person using Neural Networks with Python.

## Problem Statement

The objective of this task is we are given a data set of patient medical records of Pima Indians and whether they had an onset of diabetes within five years. 

Using these data, we will build an effective and optimised Binary Classifier and Predict the possibility of Diabetes in a person using GridSearchCV with Neural Networks and make predictions on test data.

## Dataset Description

The dataset comprises of 2 .csv files which are named “train.csv” and “test.csv”.

1.	Train.csv: Training data to use with Model Training and Validation.
- Number of entries: 668
- Attributes: 9

2.	Test.csv: Testing data to use with Predictions.
- Number of entries: 100
- Attributes: 8

![image](https://user-images.githubusercontent.com/31254745/156607971-bcd1480e-d403-434a-bbc1-51f54448fb28.png)

## Steps to Build a Neural Network Model using Keras & Optimisation with GridSearchCV

1. Importing Libraries and Loading the dataset
2. Data Exploration on all Attributes
3. Data Visualisation on all Attributes
4. Feature Scaling on all Attributes
5. Compiling the Neural Network Model
6. Hyperparameter Tuning of Neural Network Model using GridSearchCV
7. Model Building and Optimisation of Neural Network Model with Best Hyper Parameters 
8. Evaluating Model Performance on Training and Validation data
9.  Predictions on Test Data

## Hyperparameter Tuning of the Neural Network Model using GridSearchCV


Developing deep learning models is an iterative process and have lots of hyperparameters. It is very hard to tune the hyperparameter manually to get a model that can be trained efficiently in terms of time and compute resources. 

So, there is a way where we can adjust the setting of the neural networks which is called hyperparameters and the process of finding a good set of hyperparameters is called hyperparameter tuning.  

In this task, I have implemented hyperparameter tuning using GridSearchCV to build an optimised Neural Network model with better Accuracy and Performance.

- Hyperparameter Tuning "Epochs" and "Batch size"
- Hyperparameter Tuning "Learning Rate" and "Drop Out Rate"
- Hyperparameter Tuning "Activation Function" and "Kernel Initializer"
- Hyperparameter Tuning "Hidden Layer Neuron 1" & "Hidden Layer Neuron 2"

