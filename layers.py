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
		for n in lPrms:
			if hasattr(self,n):
				setattr(self,n,lPrms[n])
			else:
				raise Exception( "Attribute '%s' not found"%n )
		#The gradients wrt to the parameters and the bottom
		self.grad_ = {} 
		#Storing the weights and other stuff
		self.prms_ = {}
	
	@property
	def type(self):
		return type(self).__name__

	#Forward pass
	def forward(self, bot, top):
		pass

	#Backward pass
	def backward(self, bot, top, botgrad, topgrad):
		'''
			bot    : The inputs from the previous layer
			topgrad: The gradient from the next layer
			botgrad: The gradient to the bottom layer
		'''
		pass

	#Setup the layer including the top
	def setup(self, bot, top):
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
		if len(self.prms_) <= 0: return np.empty((0,))
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
	def setup(self, bot, top):
		for b,t in zip(bot,top):
			t.resize(b.shape,refcheck=False)

	def forward(self, bot, top):
		for b,t in zip(bot,top):
			t[...] = np.maximum(b, 0)

	def backward(self, bot, top, botgrad, topgrad):
		for b,t,db,dt in zip(bot, top, botgrad, topgrad):
			db[...] = dt * (t>0)
	
##
# Sigmoid
class Sigmoid(BaseLayer):
	'''
		f(x) = 1/(1 + exp(-sigma * x))
	'''
	sigma = 1.0
	
	def setup(self, bot, top):
		for b,t in zip(bot,top):
			t.resize(b.shape,refcheck=False)
		
	def forward(self, bot, top):
		for b,t in zip(bot,top):
			# Numerically more stable
			d = -b * self.sigma
			ep = np.exp( -np.maximum(0,d) )
			en = np.exp( np.minimum(0,d) )
			t[...] = ep / (ep + en)

	def backward(self, bot, top, botgrad, topgrad):
		for b,t,db,dt in zip(bot, top, botgrad, topgrad):
			db[...] = dt * t * (1 - t) * self.sigma

##
#Inner Product
class InnerProduct(BaseLayer):
	'''
		The input and output will be batchSz * numUnits
	'''
	##TODO: Define weight fillers
	outShape = 10
	def __init__(self, **kwargs):
		super(InnerProduct, self).__init__(**kwargs)
		try: 
			self.outShape = tuple(self.outShape)
		except: 
			self.outShape = (self.outShape,)
	
	def setup(self, bot, top):
		assert len(bot) == 1 and len(top) == 1
		top[0].resize( self.outShape, refcheck=False)
		# Initialize the parameters
		self.prms_['w'] = np.zeros(bot[0].shape + self.outShape, dtype=bot[0].dtype)
		self.prms_['b'] = np.zeros(self.outShape, dtype=bot[0].dtype)
		self.grad_['w'] = np.zeros_like(self.prms_['w'])
		self.grad_['b'] = np.zeros_like(self.prms_['b'])

	def forward(self, bot, top):
		top[0][...] = np.tensordot(bot[0],  self.prms_['w'], axes=len(bot[0].shape)) + self.prms_['b'] 

	def backward(self, bot, top, botgrad, topgrad):
		#Gradients wrt to the inputs
		botgrad[0][...]  = np.tensordot(topgrad[0], self.grad_['w'].transpose(), axes=len(self.outShape)) 
		#Gradients wrt to the parameters
		self.grad_['w'][...] = np.tensordot(bot[0], topgrad[0], axes=0).transpose()
		self.grad_['b'][...] = topgrad[0]

##
#SoftMax
class SoftMax(BaseLayer):
	def setup(self, bot, top):
		assert len(bot) == 1 and len(top) == 1
		top[0].resize( bot[0].shape, refcheck=False)
		
	def forward(self, bot, top):
		mn = np.min(bot[0])
		top[0][...] = np.exp((top[0][...] - mn))
		Z           = np.sum(top[0])
		top[0][...] = top[0] / Z

	def backward(self, bot, top, botgrad, topgrad):
		'''
			sk, pk, gj - score, probability of kth unit and jth top gradient. 
			d(output)|d(sk) = pk * sk * [\sum_j (-gj * pj) + gk]
		'''
		sm = -np.tensordot(topgrad[0], top[0], axes=len(top[0].shape)) + topgrad[0]
		botgrad[0][...] = top[0] * bot[0] * sm

##
#SoftMaxWithLoss 	
class SoftMaxWithLoss(BaseLayer):
	def setup(self, bot, top):
		assert len(bot)==2 and len(top)==1
		assert bot[0].ndim ==1, 'The bottom to the loss layer should be 1-D'
		top[0].resize((1,), refcheck=False)
		self.softmaxTop_ = np.zeros_like(bot[0])
		self.softmax_ = SoftMax([bot[0]], self.softmaxTop_)
		self.softmax_.setup([bot[0]], self.softmaxTop_)

	def forward(self, bot, top):
		lbl = bot[1]
		self.softmax_.forward([bot[0]], self.softmaxTop_)
		top[0][...] = -np.log2(self.softmaxTop_[lbl])

	def backward(self, bot, top, botgrad, topgrad):
		lbl = bot[1]
		self.botgrad[0][...] = -copy.deepcopy(self.softmaxTop_)
		self.botgrad[0][lbl] += 1 
