## @package layers
# Definition of various supported layers

import numpy as np
import scipy.misc as scm
import copy
import utils

##
# The base class from which other layers will inherit. 
class BaseLayer(object):
	def __init__(self, **lPrms):
		#The layer parameters - these can
		#be different for different layers
		print lPrms
		for n in lPrms:
			if hasattr(self,n):
				setattr(self,n,lPrms[n])
			else:
				raise Exception( "Attribute '%s' not found"%n )
		#The gradients wrt to the parameters and the bottom
		self.grad_ = {} 
		#Storing the weights and other stuff
		self.prms_ = {}
		

	#Forward pass
	def forward(self, bottom, top):
		pass

	#Backward pass
	def backward(self, bottom, top, botgrad, topgrad):
		'''
			bottom : The inputs from the previous layer
			topgrad: The gradient from the next layer
			botgrad: The gradient to the bottom layer
		'''
		pass

	#Setup the layer including the top
	def setup(self, bottom, top):
		pass

	@property
	def gradient(self):
		'''
			Get the gradient of the parameters
		'''
		return self.grad_
	@property
	def flat_gradient(self):
		'''
			Get the gradient of the parameters as a 1d array
		'''
		return np.concatenate( [self.grad_[n].ravel() for n in sorted(self.grad_)], axis=0 )

	@property
	def flat_parameters(self):
		""" Fetch all the parameters of the layer and return them as a 1d array """
		return np.concatenate( [self.prms_[n].ravel() for n in sorted(self.prms_)], axis=0 )
	@flat_parameters.setter
	def flat_parameters(self, value):
		""" Set all the parameters of the layer """
		k = 0
		for n in sorted(self.prms_):
			kk = k + self.prms_[n].size
			self.prms_[n].flat[...] = value[k:kk]
			k = kk
	
	@property
	def parameters(self):
		""" Return the layer parameters """
		return self.prms_

	def get_mutable_gradient(self, gradType):
		'''
			See get_gradient for the docs
		'''
		assert gradType in self.grad_.keys(), 'gradType: %s is not recognized' % gradType
		return copy.deepcopy(self.grad_[gradType])	

##
# Recitified Linear Unit (ReLU)
class ReLU(BaseLayer):
	type_ = 'ReLU'
	def __init__(self, **lPrms):
		super(ReLU, self).__init__(**lPrms)

	def setup(self, bottom, top):
		top = np.zeros_like(bottom)

	def forward(self, bottom, top):
		top[...] = np.maximum(bottom, 0)

	def backward(self, bottom, top, botgrad, topgrad):
		botgrad[...] = topgrad * (top>0)	
	
##
# Sigmoid
class Sigmoid(BaseLayer):
	'''
		f(x) = 1/(1 + exp(-sigma * x))
	'''
	type_ = 'Sigmoid'
	sigma = 1.0
	def __init__(self, **lPrms):
		super(Sigmoid, self).__init__(**lPrms)

	def setup(self, bottom, top):
		top = np.zeros_like(bottom)
		
	def forward(self, bottom, top):
		top[...] = 1.0 / (1 + np.exp(-bottom * self.sigma))

	def backward(self, bottom, top, botgrad, topgrad):
		botgrad[...] = topgrad * (top) * (1 - top) * self.sigma

##
#Inner Product
class InnerProduct(BaseLayer):
	'''
		The input and output will be batchSz * numUnits
	'''
	##TODO: Define weight fillers
	type_  = 'InnerProduct'
	numOut = 10	
	def __init__(self, **lPrms):
		super(InnerProduct, self).__init__(**lPrms)

	def setup(self, bottom, top):
		top = np.zeros((bottom.shape[0], self.numOut)).astype(bottom.dtype)
		self.prms_['w'] = np.zeros((bottom.shape[1], self.numOut).astype(bottom.dtype)
		self.prms_['b'] = np.zeros((1, self.numOut).astype(bottom.dtype)
		self.grad_['w'] = np.zeros_like(self.prms_['w'])
		self.grad_['b'] = np.zeros_like(self.prms_['b'])	

	def forward(self, bottom, top):
		top[...] = np.dot(bottom,  self.prms_['w']) + self.prms_['b'] 

	def backward(self, bottom, top, botgrad, topgrad):
		N = bottom.shape[0]
		#Gradients wrt to the inputs
		botgrad[...]  = np.dot(topgrad, np.transpose(self.grad_['w'])) 
		#Gradients wrt to the parameters
		self.grad_['w'][...] = np.dot(topgrad.transpose(), bottom).transpose() / N
		self.grad_['b'][...] = np.sum(topgrad, axis=0) / N

##
#SoftMax
class SoftMax(BaseLayer):
	type_ = 'SoftMax'
	def __init__(self, **lPrms):
		super(InnerProduct, self).__init__(**lPrms)

	def setup(self, bottom, top):
		top = np.zeros_like(bottom)
		
	def forward(self, bottom, top):
		top[...] = 1.0 / (1 + np.exp(-bottom * self.sigma))

	def backward(self, bottom, top, botgrad, topgrad):
		botgrad[...] = topgrad * (top) * (1 - top) * self.sigma

