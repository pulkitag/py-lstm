## @package tests
# Unit Tests
#

import numpy as np
import layers as ly

def test_relu():
	relu   = ly.ReLU()
	bottom = np.random.random((4,5))
	top    = np.zeros_like(bottom)
	 	
