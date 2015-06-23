## @package solver
# Definition of the solver module for training and testing a net

import numpy as np
import random
import scipy.optimize as sopt

class Solver:
    def __init__(self):
        # self.net_ = net
        pass

	#Train net (SGD)
	def train(self, net, data, labels, params):
        # data: list of NxD arrays. N is the sequence length (varies 
        #    for different instances)
        # labels: list of target output sequences
        
        if params.train_scratch:
		  net.init_random()
        N = len(data)
        tmp = list(zip(data, labels))
        random.shuffle(tmp)
        data, labels = zip(*tmp)
        # TODO: mini-batches?
        for i in xrange(0, N):
        	data_seq = data[i]
        	label_seq = labels[i]
            # Pass to network backward function
            net.forwardBackwardAll()
            # Compute loss
            # net.compute_loss(label_seq)
            # Backward pass
            # net.backward_all()
        	# Update weights
            grads = params.momentum * prev_grads 
                + params.learn_rate * curr_grads
            net.flat_parameters += grads
            # Snapshot the net
            if i % params.snapshot == 0:
                pass
            # Update learning rate
            if i % params.stepsize == 0:
                params.learn_rate *= params.gamma
        return net

    #Test net
   def test(self, net, data, params):
        N = len(data)
        pred = []
        for i in xrange(0, N):
   		   data_seq = data[i]
           # Reset the net for new sequence input
   		   net.reset()
   		   out = net.forward_all(input=data_seq)
   		   pred.append(out)
        return pred