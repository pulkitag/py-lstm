## @package tests
# Unit Tests
#

import numpy as np
import layers as ly
import pdb

def test_relu():
	relu   = ly.ReLU(prms={'name': 'relu1'})
	bottom = np.random.randn(4,5)
	relu.setup(bottom)
	relu.forward(bottom)
	relu.backward(bottom, np.ones_like(bottom))
	pdb.set_trace()
		
