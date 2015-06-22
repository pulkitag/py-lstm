## @package solver
# Definition of the solver module for training and testing a net

import numpy as np
import random

class Solver:
	def __init__(self):
		
	#Train net
	def train(self, net, data, labels, params):
		net.init_weights()
        N = len(data)
        tmp = list(zip(data, labels))
        random.shuffle(tmp)
        data, labels = zip(*tmp)
        for i in xrange(0, N):
        	data_seq = data[i]
        	label_seq = labels[i]
        	# Pass to network backward func
        return net

    #Test net
    def test(self, net, data, params):
    	N = len(data)
    	pred = []
    	for i in xrange(0, N):
    		data_seq = data[i]
    		net.reset()
    		out = net.forward_all(input=data_seq)
    		pred.append(out)
    	return pred

    #Gradient checking for a layer
    def check_grad_layer(self, layer, data):
    	top = np.zeros_like(lay.top_)
    	# FIXME: pass by reference in python?
    	layer.forward(bottom, top)
    	#The following check_grad() takes a function as input
    	f = lambda x : x
    	err = scipy.optimize.check_grad(f(top), 
    		layer.get_gradient(), data)
    	return err