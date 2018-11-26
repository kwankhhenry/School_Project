#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### COMP7404 Assignment 2 ##################
'''
This program builds a logistic regression classifier to 
classify whether clients default on their loan payment. 
The algorithm consists of these steps: 
(1) Loading training data from a text file into numpy matrices
(2) Transforming input features values to discrete values
(3) Computing optimal parameter values
(4) Loading testing data from a text file into numpy matrices
(5) Predicting values for testing data and evaluating performance

My submission for this assignment is entirely my own original work done 
exclusively for this assignment. I have not made and will not make my 
solutions to assignment, homework, quizzes or exams available to anyone else.
These include both solutions created by me and any official solutions provided 
by the course instructor or TA. I have not and will not engage in any 
activities that dishonestly improve my results or dishonestly improve/hurt
the results of others.

### ACKNOWLEDGMENT (Type your full name and date below)
Your full name: Kit Hueng Henry Kwan 3035420297
Date: 17 November 2018

### 
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_code import *

# This function is to load the training dataset and testing dataset (from the given .csv files)
# (train_path, test_path) -- the paths of training dataset and testing dataset
# (X_train, y_train, X_test, y_test) -- the normalized training attribute value array, training label vector, testing attribute value array, testing label vector
def load_dataset(train_path, test_path):   

    train_data = pd.read_csv(train_path, sep=',').values
    test_data = pd.read_csv(test_path, sep=',').values
    label = pd.read_csv(train_path, sep=',').columns

    train_rows = train_data.shape[0]
    test_rows = test_data.shape[0]
    
    ### START CODE HERE ###
    # s_train_array, s_test_array denote the scaled train_data and test_data
    s_train_array, s_test_array = scaleFeature(train_data, test_data)

    ### END CODE HERE ###

    # append a column of ones to the front of the datasets
    train_append_ones = np.ones([train_rows,1])
    test_append_ones = np.ones([test_rows,1])
    
    train_credit_card_data = np.column_stack((train_append_ones, s_train_array))
    test_credit_card_data = np.column_stack((test_append_ones, s_test_array))
    
    train_columns = train_credit_card_data.shape[1]                      
    test_columns = train_credit_card_data.shape[1]
    
    X_train = train_credit_card_data[:,0:train_columns-1]
    y_train = train_credit_card_data[:,train_columns-1:]
   
    X_test = test_credit_card_data[:,0:test_columns-1]
    y_test = test_credit_card_data[:,test_columns-1:]
   
    return X_train, y_train, X_test, y_test, label

# This function is to normalize the values to [0,1]
# (train_array, test_array) -- the unnormalized training array and testing array
# (s_train_array, s_test_array) -- the normalized training array and testing array
def scaleFeature(train_array, test_array):
    
    train_rows = train_array.shape[0]
    test_rows = test_array.shape[0]
    
    array_min = np.min(train_array,axis=0).reshape(1,-1)
    array_max = np.max(train_array,axis=0).reshape(1,-1)
    array_range = array_max - array_min
    
    s_train_array = np.true_divide((train_array - np.repeat(array_min,train_rows,axis=0)),np.repeat(array_range,train_rows,axis=0))
    s_test_array = np.true_divide((test_array - np.repeat(array_min,test_rows,axis=0)),np.repeat(array_range,test_rows,axis=0))
    
    return s_train_array, s_test_array


# This function is to compute the sigmoid of z (z is a scalar or numpy array of any size.), s is the result.
def sigmoid(z):
    ### START CODE HERE ###
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###
    return s

# This function is to implement the cost function and its gradient for the propagation.
# w -- weights, a numpy array,
# X -- attribute data,
# Y -- true "label" vector.
# cost -- negative log-likelihood cost for logistic regression,
# dw -- gradient of the loss with respect to w, thus same shape as w,
def propagate(w, X, Y):

    m = X.shape[0]
    
    # compute activation, A is the activation result from X and w   
    ### START CODE HERE ###
    z = X.dot(w)
    A = sigmoid(z)
    ### END CODE HERE ###
    # compute cost
    ### START CODE HERE ###
    cost = (np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))) * (-1/m)
    dw = np.dot(np.transpose(X), A-Y)*(1/m)
    ### END CODE HERE ###
    return dw, cost


# This function is to optimizes w and b by running a gradient descent algorithm.
# w -- weights, a numpy array,
# X -- attribute data,
# Y -- true "label" vector,
# num_iterations -- number of iterations of the optimization loop,
# learning_rate -- learning rate of the gradient descent update rule,
# print_cost -- True to print the loss every 100 steps.
# params -- dictionary containing the weights w,
# grads -- dictionary containing the gradients of the weights,
# costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
def optimize(w, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    file = open("costByIteration.txt","a")
    for i in range(num_iterations):  
        # Calculate the cost and gradient
        ### START CODE HERE ### 
        dw, cost = propagate(w, X, Y)
        ### END CODE HERE ###   
        # Update w 
        ### START CODE HERE ###
        w = w - learning_rate * dw
        ### END CODE HERE ###   
        # Save the cost in costs list
        costs.append(cost)
        s = 'Iteration '+str(i+1)+': '+str(costs[i])
        # Print the cost every 100 training iterations
        if print_cost and (i + 1) % 100 == 0:
            print ("Cost after iteration %i: %f" %(i + 1, cost))
        file.write(s + '\n')
   
    
    params = {"w": w}
    
    grads = {"dw": dw}

    return params, grads, costs

# This function is to predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
# w -- weights, a numpy array,
# b -- bias, a scalar,
# X -- attribute data.
# Y_prediction -- a numpy array (vector) containing all predictions for the examples in X
# def predict(w, b, X):
def predict(w, X):
    
    m = X.shape[0]
    Y_prediction = np.zeros((m,1))
    
    # compute activation, A is the activation result from X and w   
    ### START CODE HERE ###             
    z = X.dot(w)
    A = sigmoid(z)
    ### END CODE HERE ###

    for i in range(A.shape[0]):
        ### START CODE HERE ### 
        if A[i,0] <= 0.5:
            Y_prediction[i][0] = 0
        else:
            Y_prediction[i][0] = 1
        ### END CODE HERE ### 
    return Y_prediction


# This function is to build the logistic regression model by calling the previously implemented function. 
# X_train -- training set represented by a numpy array,
# Y_train -- training labels represented by a numpy array (vector),
# X_test -- test set represented by a numpy array,
# Y_test -- test labels represented by a numpy array (vector),
# num_iterations -- hyperparameter representing the number of iterations to optimize the parameters,
# learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize(),
# print_cost -- Set to true to print the cost every 100 iterations.
# d -- dictionary containing information about the model.

def model(X_train, Y_train, X_test, Y_test, label, num_iterations, learning_rate, print_cost = False):
    print('################### COMP7404 Assignment 2 #####################')
    print('#                                                             #')
    print('#  Classifying Credit Loan Default Using Logistic Regression  #')
    print('#                                                             #')
    print('###############################################################')
    print()
   
    # Print the general informations of the dataset
    m_train = X_train.shape[0]
    m_test = X_test.shape[0]
    num_px = X_train.shape[1]

    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Number of attributes: num_px = " + str(num_px))
    print ("train_set_x shape: " + str(X_train.shape))
    print ("train_set_y shape: " + str(Y_train.shape))
    print ("test_set_x shape: " + str(X_test.shape))
    print ("test_set_y shape: " + str(Y_test.shape))
    
    # Initialize parameters with zeros 
    ### START CODE HERE ###
    w = np.array([[0.0 for i in range(num_px)]]).T
    ### END CODE HERE ###
    # Gradient descent 
    parameters, grads, costs = optimize(w, X_train, Y_train, num_iterations, learning_rate, print_cost)
   
    # draw the training cost curve
    fig = plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.plot(range(num_iterations),costs,'y-',figure=fig)
    plt.savefig('./cost_curve.png')
    plt.close()
    print("Files written: cost_curve.png, costByIteration.txt")
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]

    # Predict test/train set examples
    Y_prediction_train = predict(w, train_set_x)
    Y_prediction_test = predict(w, test_set_x)
        
    # Print the labels
    print('Feature: '+str(label))
    
    # Print the parameters
    print('The optimal parameters [w] is')
    print(w)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

    return 



# Run the program on credit card dataset
if __name__ == '__main__':
    # Run the test code
    run_test(load_dataset,sigmoid,propagate,optimize,predict)
    # Loading the dataset 
    train_set_x, train_set_y, test_set_x, test_set_y,label  = load_dataset(os.getcwd() + '/credit_card_train.csv', os.getcwd() + '/credit_card_test.csv')
    # Run the model
    model(train_set_x, train_set_y, test_set_x, test_set_y, label, num_iterations = 2000, learning_rate = 0.3, print_cost = True)
	


