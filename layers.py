import numpy as np
import scipy.misc as scm

##
# The base class from which other layers will inherit. 
class BaseLayer:
	def __init__(self):
		self.type_ = None
		self.name_ = None
		self.prms_ = None

	def forward(self):
		pass

	def backward(self):
		pass

	def setup(self):
		pass
