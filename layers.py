## @package layers
# Definition of various supported layers

import numpy as np
import scipy.misc as scm

##
# The base class from which other layers will inherit. 
class BaseLayer:
	def __init__(self, prms):
		#Type of layers
		self.type_ = None
		#Name of the layers
		self.name_ = None
		#The layer parameters - these can
		#be different for different layers
		self.prms_ = None

	def forward(self, bottom, top):
		pass

	def backward(self, bottom, top):
		pass

	def setup(self, bottom, top):
		pass


class ReLU(BaseLayer):
	def setup(self, bottom, top):
		top = np.zeros_like(bottom)

	def forward(self, bottom, top):
		top = np.maximum(bottom, 0)
	
	
