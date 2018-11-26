#!/usr/bin/python
# -*- coding: utf-8 -*-

############### COMP7404 Assignment 1 ##################
'''
This program builds a linear regression predictor to 
forecast the price of houses. The algorithm consists 
of four steps: 
(1) Loading training data from a text file into numpy matrices
(2) Computing optimal parameter values
(3) Loading testing data from a text file into numpy matrices
(4) Predicting values for testing data and evaluating performance

My submission for this assignment is entirely my own original work done 
exclusively for this assignment. I have not made and will not make my 
solutions to assignment, homework, quizzes or exams available to anyone else.
These include both solutions created by me and any official solutions provided 
by the course instructor or TA. I have not and will not engage in any 
activities that dishonestly improve my results or dishonestly improve/hurt
the results of others.

### ACKNOWLEDGMENT (Type your full name and date here)
Your full name: Kit Hueng Henry Kwan (3035420297)
Date: 10 November 2018

### 
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This function is an exercise to load a file, create an array and implement arithmetic operations on array
def warmUp(warmUp_path):
    label = ['column1', 'column2', 'column3']
    warmUp_list = [[ 7.3, 2.4, 6.2],[ 5.6, 3.7, 4.8]]
    warmUp_data = pd.DataFrame(columns=label, data=warmUp_list)
    warmUp_data.to_csv('./warmUp.csv', index= False)
    # load the given warmUp.csv file in an array format
    # the expected result is [[ 7.3, 2.4, 6.2],[ 5.6, 3.7, 4.8]]
    array_A = pd.read_csv(warmUp_path, sep=',').values
    # print the label of warmUp.csv
    # the expected result is Index([u'column1', u'column2', u'column3'], dtype='object')
    label = pd.read_csv(warmUp_path, sep=',').columns
    ### START CODE HERE ###
    # creat an array array_B [[ 3.5, 4.7, 5.5],[ 4.8, 6.2, 3.9]]
    array_B = [[ 3.5, 4.7, 5.5],[ 4.8, 6.2, 3.9]]
    # calculate the transpose of array_B as transpose_B 
    # the expected result is [[ 3.5, 4.8],[ 4.7, 6.2],[ 5.5, 3.9]]
    transpose_B = np.transpose(array_B) 
    # calculate the sum of array_A and array_B as sum_AB
    # the expected result is [[ 10.8, 7.1, 11.7],[ 10.4, 9.9, 8.7]]
    sum_AB = array_A + array_B
    # calculate the difference between array_A and array_B as diff_AB
    # the expected result is [[ 3.8, -2.3, 0.7],[ 0.8, -2.5, 0.9]]
    diff_AB = array_A - array_B 
    # calculate the elementwise product of array_A and array_B as ew_product_AB
    # the expected result is [[ 25.55, 11.28, 34.1],[ 26.88, 22.94, 18.72]]
    ew_product_AB = np.multiply(array_A, array_B) 
    # calculate the array product of array_A and transpose_B as mat_product_ABt
    # the expected result is [[ 70.93, 74.1],[ 63.39, 68.54]]
    mat_product_AB = array_A.dot(transpose_B)
    # calculate the true quotient (return the float division result) between A and B as divide_AB
    # the expected result is [[ 2.08571429, 0.5106383, 1.12727273],[ 1.16666667, 0.59677419, 1.23076923]]
    divide_AB = np.divide(array_A, array_B) 
    # return the maximum of each row in array_B as max_row_B
    # the expected result is [ 5.5, 6.2]
    max_row_B = np.max(array_B, axis=1) 
    # return the minimum of each column in array_B as min_column_B
    # the expected result is [ 3.5, 4.7, 3.9]
    min_column_B = np.min(array_B, axis=0)
    # calculate the mean value of each row in array_B as mean_row_B
    # the expected result is [ 4.56666667, 4.96666667]
    mean_row_B = np.mean(array_B, axis=1) 
    ### END CODE HERE ###

    result = [{"array_A": array_A},
              {"label": label},
              {"array_B": array_B},
              {"transpose_B": transpose_B},
              {"sum_AB": sum_AB},
              {"diff_AB": diff_AB},
              {"ew_product_AB": ew_product_AB},
              {"mat_product_AB": mat_product_AB},
              {"divide_AB": divide_AB},
              {"max_row_B": max_row_B},
              {"min_column_B": min_column_B},
              {"mean_row_B": mean_row_B}]
    return result

# This function is to load the training dataset and testing dataset (from the given .csv files)
# inputs (train_path, test_path) denote the paths of training dataset and testing dataset
# append a column of ones to the front of the datasets to denote the constant vector b
# outputs (X_train, y_train, X_test, y_test) denote the normalized training attribute value array, training label vector, testing attribute value array, testing label vector
def load_dataset(train_path,test_path):
    # train_data, test_data and label denote the original training data, testing data in array format and the column labels
    train_data = pd.read_csv(train_path, sep=',').values
    test_data = pd.read_csv(test_path, sep=',').values
    label = pd.read_csv(train_path, sep=',').columns
   
    # m_train and m_test denote the number of training data and testing data
    m_train = train_data.shape[0]
    m_test = test_data.shape[0]

    ### START CODE HERE ###
    # s_train_array, s_test_array denote the scaled train_data and test_data
    #s_train_array, s_test_array = None
    train_mean = np.mean(train_data, axis=0)
    train_min = np.min(train_data, axis=0)
    train_max = np.max(train_data, axis=0)

    s_train_array = (train_data - train_mean)/(train_max - train_min)
    s_test_array = (test_data - train_mean)/(train_max - train_min)

    ### END CODE HERE ###

    # append a column of ones to the front of the datasets
    # train_append_ones and test_append_ones denote the ones vector appended to the datasets
    # train_boston_hp_data and test_boston_hp_data denote the modified training data and testing data after the appending operation
    train_append_ones = np.ones([m_train,1])
    test_append_ones = np.ones([m_test,1])
    train_boston_hp_data = np.column_stack((train_append_ones, s_train_array))
    test_boston_hp_data = np.column_stack((test_append_ones, s_test_array))
    
    # train_columns and test_columns denote the number of columns of the modified training data and testing data
    train_columns = train_boston_hp_data.shape[1]
    test_columns = test_boston_hp_data.shape[1]
    
    X_train = train_boston_hp_data[:,0:train_columns-1]
    y_train = train_boston_hp_data[:,train_columns-1:]
    
    X_test = test_boston_hp_data[:,0:test_columns-1]
    y_test = test_boston_hp_data[:,test_columns-1:]

    return X_train, y_train, X_test, y_test, label


# This function is to scale all values to [-1,1]
# inputs (train_array, test_array) denote the unscaled training array and testing array
# outputs (s_train_array, s_test_array) denote the scaled training array and testing array
def scaleFeature(train_array, test_array):
    ### START CODE HERE ###
    # m_train and m_test store the number of training and testing examples
    m_train = train_array.shape[0]
    m_test = test_array.shape[0]
    ### END CODE HERE ###
    # array_mean stores the means of the training data 
    array_mean = np.reshape(np.mean(train_array,axis=0),(1,-1))
    # array_range is difference between max and min values
    array_range = np.reshape(np.max(train_array,axis=0),(1,-1)) - np.reshape(np.min(train_array,axis=0),(1,-1))
    s_train_array = np.true_divide((train_array - np.repeat(array_mean,m_train,axis=0)),np.repeat(array_range,m_train,axis=0))
    s_test_array = np.true_divide((test_array - np.repeat(array_mean,m_test,axis=0)),np.repeat(array_range,m_test,axis=0))
    return s_train_array, s_test_array

# This function is to calculate the cost between predicted values and label values by use of MSE
# inputs (X, y, theta) denote the attribute value array, label value array and coefficient vector
# output cost is the calculated cost
def computeCost(X, y, theta):
    ### START CODE HERE ###
    m = X.shape[0]
    summands = X.dot(theta)
    cost = np.sum(np.power(summands - y, 2))/(2*m)
    ### END CODE HERE ###
    return cost

# This function is to modify the coefficient vector by use of gradient desent and save the costs after each iteration into a .txt file
# inputs (X, y, learning_rate, num_iterations) denote the attribute value array, label value array, learning rate and number of interations
# outputs (theta, cost_list) denote the modified coefficient vector and cost list contains the costs calculated after each iteration 
def gradientDescent(X, y, learning_rate, num_iterations):
    num_parameters = X.shape[1]                              
    theta = np.array([[0.0 for i in range(num_parameters)]]).T
    cost_list = [0.0 for i in range(num_iterations)]
    file = open("costByIteration.txt","a")
    ### START CODE HERE ###
    for it in range(num_iterations):
        y_predicted = X.dot(theta)
        error = (1/X.shape[0])*np.transpose(X).dot(y_predicted - y)
        theta_gradient = learning_rate*error
        theta = theta - theta_gradient
        cost_list[it] = computeCost(X, y, theta) 
    ### END CODE HERE ###   
        s = 'Iteration '+str(it+1)+': '+str(cost_list[it])
        if (it+1) % 100 == 0:
            print(s)
        file.write(s + '\n')
    return theta,  cost_list


# run the warmUp
result = warmUp(os.getcwd() + '/warmUp.csv')
for x in result:
    for key, value in x.items():
        print (key)
        print (value)

# This function is to test your written function load_dataset
def testcode_load_dataset():
    label = ['c1', 'c2', 'c3']
    train_list = [[1.,3.,4.],[6.,7.,9.]]
    test_list = [[5.,3.,5.],[2.,6.,8.]]
    train_sample = pd.DataFrame(columns=label, data=train_list)
    test_sample = pd.DataFrame(columns=label, data=test_list)
    train_sample.to_csv('./train_path.csv', index= False)
    test_sample.to_csv('./test_path.csv', index= False)
    X_train, y_train, X_test, y_test, label = load_dataset('./train_path.csv','./test_path.csv')
    if (X_train == [[ 1., -0.5, -0.5],[ 1., 0.5, 0.5]]).all() and (y_train == [[-0.5],[ 0.5]]).all() and (X_test == [[ 1., 0.3, -0.5],[ 1., -0.3, 0.25]]).all() and (y_test == [[-0.3],[ 0.3]]).all() and (label == ['c1', 'c2', 'c3']).all():
        print ('The result of load_dataset is correct.')
    else:
        print ('The result of load_dataset is wrong.')
    print ('X_train_sample')
    print (X_train)
    print ('y_train_sample')
    print (y_train)
    print ('X_test_sample')
    print (X_test)
    print ('y_test_sample')
    print (y_test)
    print ('label_sample')
    print (label)
    return 
# The expected output is 
# The result of load_dataset is correct.
# X_train_sample [[ 1., -0.5, -0.5],[ 1., 0.5, 0.5]]
# y_train_sample [[-0.5],[ 0.5]]
# X_test_sample [[ 1., 0.3, -0.5],[ 1., -0.3, 0.25]]
# y_test_sample [[-0.3],[ 0.3]]
# label_sample Index([u'c1', u'c2', u'c3'], dtype='object')
testcode_load_dataset()

# This function is to test your written function scaleFeature
def testcode_scaleFeature():
    train_array = np.array([[1.,3.,4.],[6.,7.,9.]])
    test_array = np.array([[5.,3.,5.],[2.,6.,8.]])
    s_train_array, s_test_array = scaleFeature(train_array,test_array)
    if (s_train_array == [[-0.5, -0.5, -0.5],[ 0.5, 0.5, 0.5]]).all() and (s_test_array == [[ 0.3, -0.5, -0.3 ],[-0.3, 0.25, 0.3 ]]).all():
        print ('The result of scaleFeature is correct.')
    else:
        print ('The result of scaleFeature is wrong.')
    print ('s_train_array_sample')
    print (s_train_array)
    print ('s_test_array_sample')
    print (s_test_array)
    return
# The expected output is 
# The result of scaleFeature is correct.
# s_train_array_sample [[-0.5, -0.5, -0.5],[ 0.5, 0.5, 0.5]]
# s_test_array_sample [[ 0.3, -0.5, -0.3 ],[-0.3, 0.25, 0.3 ]]
testcode_scaleFeature()

# This function is to test your written function computeCost
def testcode_computeCost():
    X = np.array([[1.,2.,2.],[1.,5.,3.]])
    y = np.array([[3.],[2.]])
    theta = np.array([[0.1],[0.2],[0.3]])
    cost = computeCost(X, y, theta)
    if cost ==  0.9025:
        print ('The result of computeCost is correct.')
    else:
        print ('The result of computeCost is wrong.')
    print ('cost_sample')
    print (cost)
    return
# The expected output is 
# The result of computeCost is correct.
# cost_sample  0.9025
testcode_computeCost()

# This function is to test your written function gradient Descent
def testcode_gradientDescent():
    X = np.array([[1.,1.,1.,1.],[1.,2.,2.,2.]])
    y = np.array([[1.],[2.]])
    alpha = 0.01
    num_iter = 5
    theta, cost_list = gradientDescent(X, y, alpha, num_iter)
    if (abs(theta -[[0.06327816],[0.10569322],[0.10569322],[0.10569322]]) <= 0.0000001).all() and (abs(np.array(cost_list) - np.array([1.04883125, 0.8800788490625, 0.7385191503715038, 0.6197702579899017, 0.5201563851194574])) <= 0.000000001).all():
        print ('The result of gradientDescent is correct.')
    else:
        print ('The result of gradientDescent is wrong.')
    print ('theta_sample')
    print (theta)
    print ('cost_list_sample')
    print (cost_list)
    return
# The expected output is 
# theta_sample [[0.06327816],[0.10569322],[0.10569322],[0.10569322]]
# cost_list_sample [1.04883125, 0.8800788490625, 0.7385191503715038, 0.6197702579899017, 0.5201563851194574]
testcode_gradientDescent()



# load the dataset
X_train, y_train, X_test, y_test, label = load_dataset(os.getcwd() + '/boston_train.csv', os.getcwd() + '/boston_test.csv')

# print the general informations of the dataset
print ("Number of training examples: m_train = " + str(X_train.shape[0]))
print ("Number of testing examples: m_test = " + str(X_test.shape[0]))
print ("train_set_x shape: " + str(X_train.shape))
print ("train_set_y shape: " + str(y_train.shape))
print ("test_set_x shape: " + str(X_test.shape))
print ("test_set_y shape: " + str(y_test.shape))

# training by use of gradient descent
num_iter = 3000
alpha = 0.01
theta,  cost_train = gradientDescent(X_train, y_train, alpha, num_iter)
# calculate the testing cost
cost_test = computeCost(X_test, y_test, theta)

# draw the training cost curve
fig = plt.figure()
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.plot(range(num_iter),cost_train,'y-',figure=fig)
#plt.show()
plt.savefig('./cost_curve.png')
print("Costs saved: 'cost_curve.png, costByIteration.txt'.")
plt.close()

# print the final coefficient vector, training cost, testing cost, training accuracy and testing accuracy
print('Label: '+str(label))
print('Theta:')
print(theta)

print('Best training Cost = ', cost_train[-1])
print('Testing Cost = ', cost_test)
