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
	

class A:
	a,b = 0,1
	def __init__(self,**kwargs):
		for n in kwargs:
			if hasattr(self,n):
				setattr(self,n,kwargs[n])
			else:
				print( "Attribute '%s' not found"%n )
	def __str__(self):
		return "A(%d, %d)"%( self.a, self.b )

class B(A):
	c,d = 0,1
	
a1 = A(a=10)
a2 = A(b=10)
b2 = B(a=1,b=1,c=1)

print( a1 )
print( a2 )
