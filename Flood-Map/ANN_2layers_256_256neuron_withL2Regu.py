import sys
sys.modules[__name__].__dict__.clear()

import math
import numpy as np
#import tables
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.python.framework import ops
#import matlplotlib.pyplot as plt

#%matplotlib inline

lines_x = open('Training.dat').read().splitlines()
x_data = np.zeros((5, 490))

for idx, line in enumerate(lines_x):
    x_data[0, idx] = float(line[0:8])
    x_data[1, idx] = float(line[8:17])
    x_data[2, idx] = float(line[17:26])
    x_data[3, idx] = float(line[26:35])
    x_data[4, idx] = float(line[35:]) ## Do not mix space and tab
	
X = x_data
X = X.astype('float32', casting = 'same_kind')

y_data = loadmat('WL_71212.mat')
Y = np.transpose(y_data['WL'])

def separate_train_test(X, Y, train_size):
	"""
	X is the training.dat data, which has the shape [5, 150]
	Y is the WL response data, which has the shape [10, 150], this 10 is changable.
	train_size is how many storm to feed into training set, for new set it to 120.
	Num_train_storm = 120, num_test_storm = 30.
	"""
	
	seed = 1  ## To keep it constant for all the tests.
	
	m = X.shape[1]
	permutation = list(np.random.permutation(m))
	
	X_train = X[:, :train_size]
	X_test = X[:, train_size:]
	Y_train = Y[:, :train_size]
	Y_test = Y[:, train_size:]
	
	return (X_train, Y_train, X_test, Y_test)
	
(X_train, Y_train, X_test, Y_test) = separate_train_test(X, Y, 400)


def create_placeholder(n_x, n_y):
	"""
	n_x = 5
	n_y = 37
	"""
	
	X = tf.placeholder(shape = [n_x, None], dtype = tf.float32)
	Y = tf.placeholder(shape = [n_y, None], dtype = tf.float32)
	
	return X, Y
	
def initialize_parameters():
	"""
	W1.shape = [32, 5]
	b1.shape = [32, 1]
	W2.shape = [64, 32]
	b2.shape = [64, 1]
	W3.shape = [1, 64]
	b3.shape = [1, 1]
	"""
	
	W1 = tf.get_variable("W1", [256, 5], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [256, 1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [256, 256], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [256, 1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [71212, 256], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3", [71212, 1], initializer = tf.zeros_initializer())
	
	parameters = {"W1": W1,
				  "b1": b1,
				  "W2": W2,
				  "b2": b2,
				  "W3": W3,
				  "b3": b3}
				  
	return parameters
	
def forward_propagation(X, parameters):
	"""
	"""
	
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']
	
	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)
	
	return Z3
	
def compute_cost(Z3, Y, parameters):
	"""
	"""
	# Not sure this is necessary
	#logits = tf.transpose(Z2)
	#labels = tf.transpose(Y)
	
	cost = tf.reduce_mean(tf.losses.mean_squared_error(Z3, Y))
	
	# cost for L2 regulation
	
	#W1 = parameters['W1']
	#W2 = parameters['W2']
	#W3 = parameters['W3']
	
	#cost = tf.reduce_mean(tf.losses.mean_squared_error(Z3, Y) + 0.01*tf.nn.l2_loss(W1) + 0.01*tf.nn.l2_loss(W2) + 0.01*tf.nn.l2_loss(W3))
	
	return cost
	
def random_mini_batches(X, Y, mini_batch_size = 32):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
	
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
		  num_epochs = 1000, minibatch_size = 32, print_cost = True):
		  
	"""
	Number of training examples =  400
	Number of test examples = 90
	"""
	
	ops.reset_default_graph()
	(n_x, m) = X_train.shape
	n_y = Y_train.shape[0]
	costs = []
	
	X, Y = create_placeholder(n_x, n_y)
	
	parameters = initialize_parameters()
	
	Z3 = forward_propagation(X, parameters)
	
	cost = compute_cost(Z3, Y, parameters)
	
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
	
	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		
		sess.run(init)
		
		for epoch in range(num_epochs):
			
			epoch_cost = 0.
			num_minibatches = int(m / minibatch_size)
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
			
			for minibatch in minibatches:
				
				(minibatch_X, minibatch_Y) = minibatch
				
				_, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
				
				epoch_cost += minibatch_cost / num_minibatches
				
			if print_cost == True and epoch % 100 == 0:
				print ("Cost after epoch %i: %f" %(epoch, epoch_cost))
			if print_cost == True and epoch % 5 == 0:
				costs.append(epoch_cost)
				
		#plt.plot(np.squeeze(costs))
		#plt.ylabel('cost')
		#plt.xlabel('iterations (per tens)')
		#plt.title("Learning rate =" + str(learning_rate))
		#plt.show()
		
		parameters = sess.run(parameters)
		
		prediction_difference = tf.abs(tf.subtract(Z3, Y))
		
		accuracy = tf.reduce_mean(prediction_difference)
		
		print ("Training Accuracy:" + str(accuracy.eval({X: X_train, Y: Y_train})))
		print ("Test Accuracy:" + str(accuracy.eval({X: X_test, Y: Y_test})))
		
		return parameters
		
parameters = model(X_train, Y_train, X_test, Y_test)

### Without L2 Regularization, RMSE(train) = 0.05 ; RMSE(test) = 0.14

### With L2 Regularization (0.01, 0.01, 0.001), RMSE(train) =   ; RMSE(test) =  

Z2 = forward_propagation(X_train, parameters)

with tf.Session() as sess:
  Y_train_ANN = (sess.run(Z2))
  
Z3 = forward_propagation(X_test, parameters)

with tf.Session() as sess:
  Y_test_ANN = (sess.run(Z3))
  
file = open('Y_train_ANN.dat','w+')
for i in range(0, Y_train_ANN.shape[1]):
	tmp = float(np.squeeze(Y_train_ANN[:,i]))
	file.write("%.2f" % tmp)
	file.write("\n")
file.close()

file = open('Y_test_ANN.dat','w+')
for i in range(0, Y_test_ANN.shape[1]):
	tmp = float(np.squeeze(Y_test_ANN[:,i]))
	file.write("%.2f" % tmp)
	file.write("\n")
file.close()

file = open('Y_test.dat','w+')
for i in range(0, Y_test.shape[1]):
	tmp = float(np.squeeze(Y_test[:,i]))
	file.write("%.2f" % tmp)
	file.write("\n")
file.close()

file = open('Y_train.dat','w+')
for i in range(0, Y_train.shape[1]):
	tmp = float(np.squeeze(Y_train[:,i]))
	file.write("%.2f" % tmp)
	file.write("\n")
file.close()

























