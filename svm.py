'''
Author: Harshitha Machiraju
Date  : 20/11/2018
Title : Implementation of linear SVM
Description: The program implements SVM with a linear kernel.
			->The program solves both the primal and dual form of the SVM optimization
			  and returns the solution.
			->The program uses the library cvxpy for optimization. 
			->The program uses the pandas library for reading the csv file.

'''

import numpy as np
import pandas as pd
from cvxpy import *
import matplotlib.pyplot as plt

# FUNCTION TO SOLVE THE DUAL PROBLEM
def dual_prob(X,t):
	N=X.shape[0]
	f=X.shape[1]

	alpha=Variable(N)
	sum_alpha=sum_entries(alpha)
	# Constraints
	alpha_constraint=sum_entries(mul_elemwise(t,alpha))

# 	FORMING THE QUADRATIC FORM OF THE EXPRESSION
	K_mat = np.dot(X,X.T)
	t_hat = np.multiply(t,t.T)
	Q_hat=np.multiply(K_mat,t_hat)	
	Q=quad_form(alpha,Q_hat)	

# 	OPTIMIZATION
	obj=Maximize(sum_alpha-mul_elemwise(0.5,Q))
	Constraints=[alpha_constraint==0,alpha>=0]
	prob=Problem(obj,Constraints)

	prob.solve()

# 	FINDING BETA
	beta=np.zeros((1,f))
	for k in range(N):
		mul=t[k,0]*alpha.value[k]
		beta=beta+X[k,:]*mul[0,0]

# 	FINDING BETA_0
	beta_0=[]
	for i in range(N):
		beta_0+=[1/float(t[i,0])-np.dot(beta,X[i,:])]

# MINIMUM OF ALL BETA_0 IN ABSOLUTE SENSE IS CHOSEN TO SATISFY THE PROBLEM
	d=beta_0[:]
	beta_0_abs=np.min(np.abs(d))

	for i in range(len(beta_0)):
		if((beta_0[i]==beta_0_abs) or (beta_0[i]==-1*beta_0_abs)):
			# print beta_0[i]
			beta_0_final=beta_0[i]

	return beta.T,beta_0_final


# FUNCTION TO SOLVE THE PRIMAL PROBLEM
def primal_prob(X,t):
	# Primal form
	N=X.shape[0]
	f=X.shape[1]

	beta=Variable(f)
	b_0=Variable()

# 	CONSTRAINTS FOR THE OPTIMIZATION
	Constraints=[]
	for i in range(N):
		Constraints+=[(mul_elemwise(t[i,:], X[i,:]*beta+b_0)-1)>=0]
	prob = Problem(Minimize(0.5* norm(beta, 2)),Constraints)
	prob.solve()


	return beta.value,b_0.value


# FUNCTION TO PREDICT THE CLASS USING BETA AND BETA_0
def predict_svm(beta,beta_0,x_test):
	return np.sign(np.dot(x_test,beta)+beta_0)


#Take the inputs 'X' and the labels 't'
# NOTE: THE INPUT OBSERVATIONS ARE IN ROWS
X = np.asarray(pd.read_csv('X.csv', sep=',',header=None))
t=np.asarray(pd.read_csv('t.csv', sep=',',header=None))

# Test Cases
x_1=np.array([[2,0.5]])
x_2=np.array([[-0.8,-0.7]])
x_3=np.array([[1.58,-1.33]])
x_4=np.array([[-0.008,0.001]])


beta_dual,beta_0_dual=dual_prob(X,t)
beta_primal,beta_0_primal=primal_prob(X,t)


print "------------------------------------------------------"

print "Predictions by primal optimization"
print "Value of beta: ",beta_primal
print "Value of beta_0: ",beta_0_primal
print " Class of ",x_1,predict_svm(beta_primal,beta_0_primal,x_1)
print " Class of ",x_2,predict_svm(beta_primal,beta_0_primal,x_2)
print " Class of ",x_3,predict_svm(beta_primal,beta_0_primal,x_3)
print " Class of ",x_4,predict_svm(beta_primal,beta_0_primal,x_4)

print "------------------------------------------------------"

print "Predictions by dual optimization"
print "Value of beta: ",beta_dual
print "Value of beta_0: ",beta_0_dual
print " Class of ",x_1,predict_svm(beta_dual,beta_0_dual,x_1)
print " Class of ",x_2,predict_svm(beta_dual,beta_0_dual,x_2)
print " Class of ",x_3,predict_svm(beta_dual,beta_0_dual,x_3)
print " Class of ",x_4,predict_svm(beta_dual,beta_0_dual,x_4)

print "------------------------------------------------------"