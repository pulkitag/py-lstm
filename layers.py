## @package layers
# Definition of various supported layers

import numpy as np
import scipy.misc as scm
import copy

##
# The base class from which other layers will inherit. 
class BaseLayer:
	def __init__(self, prms={'name': 'default'}):
		#Type of layers
		self.type_ = None
		#Name of the layers
		self.name_ = prms['name']
		#The layer parameters - these can
		#be different for different layers
		self.prms_ = prms
		#The output of the layer
		self.top_  = None
		#The gradients wrt to the parameters and the bottom
		self.grad_ = {} 

	#Forward pass
	def forward(self, bottom, top):
		pass

	#Backward pass
	def backward(self, bottom, topgrad):
		'''
			bottom: The inputs from the previous layer
			top   : The gradient from the next layer
		'''
		pass

	#Setup the layer including the top
	def setup(self, bottom, top):
		pass

	def get_gradient(self, gradType='bot'):
		'''
			gradType: bot - gradients with respect to the bottom inputs
								in general it can be the name of parameters wrt to which
								gradient is required
		'''
		assert gradType in self.grad_.keys(), 'gradType: %s is not recognized' % gradType
		return copy.deepcopy(self.grad_[gradType])	

	def get_mutable_gradient(self, gradType):
		'''
			See get_gradient for the docs
		'''	
		assert gradType in self.grad_.keys(), 'gradType: %s is not recognized' % gradType
		return copy.deepcopy(self.grad_[gradType])	

##
# Recitified Linear Unit (ReLU)
class ReLU(BaseLayer):
	def setup(self, bottom, top):
		top               = np.zeros_like(bottom)
		self.grad_['bot'] = np.zeros_like(bottom) 
		self.type_        = 'ReLU'

	def forward(self, bottom, top):
		top = np.maximum(bottom, 0)

	def backward(self, bottom, topgrad):
		self.grad_['bot'] = topgrad * (self.top_>0)	
	
##
# Sigmoid
class Sigmoid(BaseLayer):
	
	
