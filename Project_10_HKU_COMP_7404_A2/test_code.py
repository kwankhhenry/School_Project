#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

# This function is to test your written function load_dataset
# The expected output is 
# The result of load_dataset is correct.
# X_train_sample [[1., 0., 0.],[1., 1., 1.]]
# y_train_sample [[0.],[ 1.]]
# X_test_sample [[1., 0.2, 0.33333333],[1., 0.2, 0.66666667]]
# y_test_sample [[1.],[ 0.]]
# label_sample Index([u'c1', u'c2', u'c3'], dtype='object')
def testcode_load_dataset(func_load):
	label = ['c1', 'c2', 'c3']
	train_list = [[1.,3.,0.],[6.,6.,1.]]
	test_list = [[2.,4.,1.],[2.,5.,0.]]
	train_sample = pd.DataFrame(columns=label, data=train_list)
	test_sample = pd.DataFrame(columns=label, data=test_list)
	train_sample.to_csv('./train_path.csv', index= False)
	test_sample.to_csv('./test_path.csv', index= False)
	X_train, y_train, X_test, y_test, label = func_load('./train_path.csv','./test_path.csv')

	if (X_train == [[1., 0., 0.],[1., 1., 1.]]).all() and (y_train == [[0.],[ 1.]]).all() and (np.around(X_test, decimals=5) == [[1., 0.2, 0.33333],[1.,  0.2,  0.66667]]).all() and (y_test == [[1.],[ 0.]]).all() and (label == ['c1', 'c2', 'c3']).all():
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

# This function is to test your written function sigmoid
# The expected output is 
# The result of sigmoid is correct.
# s_sample [0.73105858, 0.88079708, 0.95257413]
def testcode_sigmoid(func_sig):
	z = np.array([1.,2.,3.])
	s = func_sig(z)
	
	if (np.around(s, decimals=5) == [0.73106, 0.88080, 0.95257]).all():
		print ('The result of sigmoid is correct.')
	else:
		print ('The result of sigmoid is wrong.')

	print ("s_sample")
	print (s)
	return

# This function is to test your written function propagate
# The expected output is 
# The result of propagate is correct.
# dw_sample [[ 0.33106399],[-0.0030299 ],[-0.0030299 ]]
# cost_sample 0.5546321427436187
def testcode_propagate(func_prop):
	w = np.array([[0.7],[1.2],[3.2]])
	X = np.array([[1., 0., 0.],[1., 1., 1.]])
	Y = np.array([[0.],[ 1.]])
	dw, cost = func_prop(w, X, Y)
	if (np.around(dw, decimals=5) == [[ 0.33106],[-0.00303],[-0.00303]]).all() and (round(cost,5) == 0.55463):
		print ('The result of propagate is correct.')
	else:
		print('The result of propagate is wrong.')
    
	print ('dw_sample')
	print (dw)
	print ('cost_sample')
	print (cost)
	return

# This function is to test your written function optimize
# The expected output is 
# The result of optimize is correct.
# parameters_sample[[0.23761509],[1.20552205],[3.20552205]]
# gradients_sample [[ 0.28570015],[-0.00436705],[-0.00436705]]
# costs_sample [0.5546321427436187, 0.5223131073276809, 0.49221473569274354, 0.464243751901144, 0.43829653727012274]
def testcode_optimize(func_opt):
	w = np.array([[0.7],[1.2],[3.2]])
	X = np.array([[1., 0., 0.],[1., 1., 1.]])
	Y = np.array([[0.],[ 1.]])
	params, grads, costs = func_opt(w, X, Y, 5, 0.3, print_cost = False)
	parameters = params["w"]
	gradients = grads["dw"]
	if (np.around(parameters, decimals=5) == [[ 0.23762],[1.20552],[3.20552]]).all() and (np.around(gradients, decimals=5) == [[ 0.28570],[-0.00437],[-0.00437]]).all() and (np.around(np.array(costs),decimals=5) == [0.55463, 0.52231, 0.49221, 0.46424, 0.43830]).all():
		print('The result of optimize is correct.')
	else:
		print('The result of optimize is wrong')

	print ("parameters_sample")
	print (parameters)
	print ("gradients_sample")
	print (gradients)
	print ("costs_sample")
	print (costs)
	return

# This function is to test your written function predict
# The expected output is 
# The result of predict is correct.
# Y_prediction_sample [[0.],[1.]]
def testcode_predict(func_pred):
	w = np.array([[-0.7],[1.2],[3.2]])
	X = np.array([[1., 0., 0.],[1., 1., 1.]])
	Y_prediction = func_pred(w, X)
	if (Y_prediction == [[0.],[1.]]).all():
		print ('The result of predict is correct.')
	else:
		print ('The result of predict is wrong.')

	print ("Y_prediction_sample")
	print (Y_prediction)
	return

def run_test(func_load,func_sig,func_prop,func_opt,func_pred):
	print("============== Code Testing ==============")
	testcode_load_dataset(func_load)
	print("==========================================")
	testcode_sigmoid(func_sig)
	print("==========================================")
	testcode_propagate(func_prop)
	print("==========================================")
	testcode_optimize(func_opt)
	print("==========================================")
	testcode_predict(func_pred)
	print("==========================================")
	return




