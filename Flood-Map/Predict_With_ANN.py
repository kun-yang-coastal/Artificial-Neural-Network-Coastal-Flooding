import math
import numpy as np
import pickle
#import tables
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.python.framework import ops
#import matlplotlib.pyplot as plt 

f = open('4layers_256_128_64_32neurons_parameters.pckl', 'rb')
parameters = pickle.load(f)
f.close()

lines_charley = open('Storm_Charley.dat').read().splitlines()
charley_data = np.zeros((5, 1))

for idx, line in enumerate(lines_charley):
	charley_data[idx, 0] = float(line[:])
	
X_charley = charley_data
X_charley = X_charley.astype('float32', casting = 'same_kind')

def forward_propagation(X, parameters):
	"""
	"""
	
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']
	W4 = parameters['W4']
	b4 = parameters['b4']
	W5 = parameters['W5']
	b5 = parameters['b5']
	
	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)
	A3 = tf.nn.relu(Z3)
	Z4 = tf.add(tf.matmul(W4, A3), b4)
	A4 = tf.nn.relu(Z4)
	Z5 = tf.add(tf.matmul(W5, A4), b5)
	
	return Z5
	
Z3 = forward_propagation(X_charley, parameters)

with tf.Session() as sess:
  Y_Charley_ANN = (sess.run(Z3))
  
file = open('Y_Charley_ANN.dat','w+')
for i in range(0, Y_Charley_ANN.shape[0]):
	tmp = float(np.squeeze(Y_Charley_ANN[i,0]))
	file.write("%.2f" % tmp)
	file.write("\n")
file.close()

#############################################################

lines_irma = open('Storm_Irma.dat').read().splitlines()
irma_data = np.zeros((5, 1))

for idx, line in enumerate(lines_irma):
	irma_data[idx, 0] = float(line[:])
	
X_irma = irma_data
X_irma = X_irma.astype('float32', casting = 'same_kind')

Z3 = forward_propagation(X_irma, parameters)

with tf.Session() as sess:
  Y_Irma_ANN = (sess.run(Z3))
  
file = open('Y_Irma_ANN.dat','w+')
for i in range(0, Y_Irma_ANN.shape[0]):
	tmp = float(np.squeeze(Y_Irma_ANN[i,0]))
	file.write("%.2f" % tmp)
	file.write("\n")
file.close()

##############################################################

lines_wilma = open('Storm_Wilma.dat').read().splitlines()
wilma_data = np.zeros((5, 1))

for idx, line in enumerate(lines_wilma):
	wilma_data[idx, 0] = float(line[:])
	
X_wilma = wilma_data
X_wilma = X_wilma.astype('float32', casting = 'same_kind')

Z3 = forward_propagation(X_wilma, parameters)

with tf.Session() as sess:
  Y_Wilma_ANN = (sess.run(Z3))
  
file = open('Y_Wilma_ANN.dat','w+')
for i in range(0, Y_Wilma_ANN.shape[0]):
	tmp = float(np.squeeze(Y_Wilma_ANN[i,0]))
	file.write("%.2f" % tmp)
	file.write("\n")
file.close()






















