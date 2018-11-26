#!/usr/bin/env python
# -*- coding: utf-8 -*-

############### COMP7404 Assignment 3 ##################
'''
This program builds a naive bayes (NB) classifier to 
classify between positive and negative reviews on books.

The algorithm consists of these steps: 
(1) Initializing the prior and likelihood
(2) Computing the probability of prior = prob( doc | class )
(3) Computing the probability of likelihood = prob( word | class )
(4) Predicting values for training and testing data, and evaluating performance

The program also compares NB with two other algorithms, 
Support Vector Machine (SVM) and Neural Network (NN), which
are to be implemented using the scikit-learn package.


My submission for this assignment is entirely my own original work done 
exclusively for this assignment. I have not made and will not make my 
solutions to assignment, homework, quizzes or exams available to anyone else.
These include both solutions created by me and any official solutions provided 
by the course instructor or TA. I have not engaged and will not engage in any 
activities that dishonestly improve my results or dishonestly improve/hurt
the results of others.

### ACKNOWLEDGMENT (Type your full name and date below)
Your full name: Kwan Kit Hueng Henry (3035420297)
Date: 22 November 2018

### 
'''

from sentiment_reader import SentimentCorpus
from multinomial_naive_bayes import MultinomialNaiveBayes
from sklearn.metrics import classification_report, confusion_matrix
import svm
import nn
if __name__ == '__main__':
    print('##################### COMP7404 Assignment 3 #####################')
    print('#                                                               #')
    print('#   Classifying Online Review Sentiment with Machine Learning   #')
    print('#                                                               #')
    print('#################################################################')
    print()

    dataset = SentimentCorpus()
    nb = MultinomialNaiveBayes()
    
    params = nb.train(dataset.train_X, dataset.train_y)
    
    predict_train = nb.test(dataset.train_X, params)
    eval_train = nb.evaluate(predict_train, dataset.train_y)
    
    predict_test = nb.test(dataset.test_X, params)
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    print("\n=======================================================\n")
    print("+++ Naive Bayes +++")
    print  ("Accuracy on training data = %f \n Accuracy on testing data = %f" % (eval_train, eval_test))
    print("Confusion Matrix:")
    print(confusion_matrix(dataset.test_y,predict_test))
    print(classification_report(dataset.test_y,predict_test))
    print("=======================================================\n")
    print("+++ Support Vector Machine +++")
    svm.run_svm(dataset.train_X, dataset.train_y, dataset.test_X, dataset.test_y)
    print("=======================================================\n")
    print("+++ Neural Network +++")
    nn.run_nn(dataset.train_X, dataset.train_y, dataset.test_X, dataset.test_y)
    print("=======================================================")
     

