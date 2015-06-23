## @package tests
# Unit Tests
#

import numpy as np
import layers as ly
import pdb

def test_relu():
	relu   = ly.ReLU()
	bottom = np.random.randn(4,5)
	top    = np.zeros_like(bottom)
	relu.setup(bottom, top)
	relu.forward(bottom, top)
	topgrad, botgrad = np.zeros_like(top), np.zeros_like(bottom)
	relu.backward(bottom, top, botgrad, topgrad)
	#pdb.set_trace()

def test_sigmoid():
	sig = ly.Sigmoid(**{'sigma':2})
	pdb.set_trace()	

