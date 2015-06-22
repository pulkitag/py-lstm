## @package layers
# Definition of various supported layers

import numpy as np
import scipy.misc as scm
import copy
import utils

##
# The default layer parameters
def get_layer_prms(layerType, ipPrms):
	prms = {}
	prms['type'] = layerType
	if layerType == 'ReLU':
		pass
	elif layerType == 'Sigmoid':
		prms['sigma'] = 1.0
	else:
		raise Exception('layerType %s not recognized' % layerType)
	newPrms = utils.update_defaults(ipPrms, prms) 
	return newPrms

##
# The base class from which other layers will inherit. 
class BaseLayer:
	def __init__(self):
		#The layer parameters - these can
		#be different for different layers
		self.prms_ = {}
		#The gradients wrt to the parameters and the bottom
		self.grad_ = {} 

	#Forward pass
	def forward(self, bottom, top):
		pass

	#Backward pass
	def backward(self, bottom, topgrad, botgrad):
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
	def __init__(self, **prms):
		super(ReLU, self).__init__()
		self.prms_ = get_layer_parameters('ReLU', prms) 

	def setup(self, bottom, top):
		top = np.zeros_like(bottom)

	def forward(self, bottom, top):
		top = np.maximum(bottom, 0)

	def backward(self, bottom, topgrad, botgrad):
		botgrad = topgrad * (self.top_>0)	
	
##
# Sigmoid
class Sigmoid(BaseLayer):
	def __init__(self, **prms):
		super(Sigmoid, self).__init__()
		self.prms_ = get_layer_parameters('Sigmoid', prms) 
	
