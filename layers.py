## @package layers
# Definition of various supported layers

import numpy as np
import scipy.misc as scm
import copy
import utils
from easydict import EasyDict as edict

##
# The base class from which other layers will inherit. 
class BaseLayer(object):
	loss_weight = 0
	def __init__(self, **lPrms):
		#The layer parameters - these can
		#be different for different layers
		for n in lPrms:
			if hasattr(self,n):
				setattr(self,n,lPrms[n])
			else:
				raise Exception( "Attribute '%s' not found"%n )
		#The gradients wrt to the parameters and the bottom
		self.grad_ = edict() 
		#Storing the weights and other stuff
		self.prms_ = edict()
	
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
		if len(self.grad_) <= 0: return np.empty((0,))
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
	
	@classmethod
	def from_self(cls, other):
		return cls(other.sigma)

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
	opSz = 10
	def setup(self, bot, top):
		assert len(bot) == 1 and len(top) == 1
		top[0].resize( self.opSz, refcheck=False)
		# Initialize the parameters
		self.prms_['w'] = np.zeros((bot[0].size,self.opSz), dtype=bot[0].dtype)
		self.prms_['b'] = np.zeros(self.opSz, dtype=bot[0].dtype)
		self.grad_['w'] = np.zeros_like(self.prms_['w'])
		self.grad_['b'] = np.zeros_like(self.prms_['b'])

	def forward(self, bot, top):
		top[0][...] = np.dot(bot[0].ravel(), self.prms_['w']) + self.prms_['b'] 

	def backward(self, bot, top, botgrad, topgrad):
		#Gradients wrt to the inputs
		botgrad[0].flat[...]  = np.dot(self.grad_['w'], topgrad[0]) 
		#Gradients wrt to the parameters
		self.grad_['w'][...] = bot[0].ravel()[:,None] * topgrad[0][None,:]
		self.grad_['b'][...] = topgrad[0]

##
#SoftMax
class SoftMax(BaseLayer):
	def setup(self, bot, top):
		assert len(bot) == 1 and len(top) == 1
		top[0].resize( bot[0].shape, refcheck=False)
		
	def forward(self, bot, top):
		#For numerical stability subtract the max
		mx = np.max(bot[0])
		top[0][...] = np.exp((top[0][...] - mx))
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

##
#LSTM Unit
class LSTM(BaseLayer):
	'''
		opSz     : The output sisze of the LSTM layer - (i.e. number of memory units)
		isLateral: False (Default) - memory cells operate independently of each other
						   True - memory cells interact with other to influence the ip/op etc. 

		Differences from the standard derivation
			- no tanH
			- One thing that seems weird is that inputs from both the memory and the 
				previous output are used to update the new state. In this version of LSTM
				I have ommited this idea. 
	'''
	opSz         = 10	
	isLateral    = False
	nonLinearityType = Sigmoid
	nonLinearityPrms = {} 
	def setup(self, bot, top):
		assert len(bot) == 1 and len(top) == 1
		assert bot[0].ndim==1
		assert type(opSz) == int
		ipSz, dType = len(bot[0]), bot[0].dtype
		top[0].resize((opSz,), refcheck=False)
		#The non-linearities
		self.nl_    = edict()
		#There are 4 modules within the lstm
		#ip, ipg, opg, fg - input, input gate, op gate and forget gate
		self.modules_ = ['ip', 'ipg', 'opg', 'fg']
		self.prms_.w, self.prms_.b = edict()
		for md in self.modules_:
			if isLateral:
				#ipSz + opSz to account for inputs from all memory units
				self.prms_.w[md] = np.zeros((opSz, ipSz + opSz), dtype=dType)
			else:
				#ipSz + 1 to account for the self input
				self.prms_.w[md] = np.zeros((opSz, ipSz + 1), dtype=dType)
			self.prms_.b[md]   = np.zeros((1, opSz), dtype=dType)	
			self.nl_[md]       = edict() 
			self.nl_[md].layer = self.nonLinearityType(**self.nonLinearityPrms)
			self.nl_[md].top   = np.zeros((opSz,), dtype=dType)
			self.nl_[md].bot   = np.zeros((ipSz,), dtype=dType)
		#The memory
		self.mem_   = np.zeros((opSz,)), dtype=dType)	
		self.memt0_ = np.zeros((opSz,)), dtype=dType)	
		if isLateral:
			self.bot_ = np.zeros((ipSz + opSz,), dtype=dType)
		else:
			self.bot_ = np.zeros((ipSz + 1,), dtype=dType)		
		self.ipSz = ipSz	

	def forward(self, bot, top):
		for md in self.modules_:
			if isLateral:
				self.bot_[0:ipSz] = bot[0][...]
				self.bot_[ipSz:]  = self.mem_[...]
				self.nl_[md].bot  = np.dot(self.prms_.w[md], self.bot_) + self.prms_.b[md] 
				self.nl_[md].layer.forward(self.bot_, self.nl_[md].top)
			else:
				pass

		#Update the memory
		self.mem_    = self.mem_ * self.nl_.fg.top + self.nl_.ipg.top * self.nl_.ip.top
		top          = self.nl_.opg.top * self.memt0_
		#Copy the current to previous state
		self.memt0_  = copy.deepcopy(self.mem_)
