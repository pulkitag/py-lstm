import numpy as np

class Network:
	def __init__(self):
		# Stores layers in the format (layer_object,input_names,output_names)
		self._layers = []
		self._blobs = {}
		self._diffs = {}
		self._blob_history = []
	
	def _get_blobs(self,names):
		return [self._blobs[n] for n in names]
	
	def _get_diffs(self,names):
		return [self._diffs[n] for n in names]
	
	def addLayer(self,layer_name,layer_instance,input_blobs,output_blobs):
		"""
		    Add a new layer_instance with inputs and outputs given as a list of strings
		"""
		if not isinstance(input_blobs ,list): input_blobs  = [input_blobs]
		if not isinstance(output_blobs,list): output_blobs = [output_blobs]
		self._layers.append( (layer_name,layer_instance,input_blobs,output_blobs) )
	
	def setup(self,**kwargs):
		"""
		    Setup the network
		"""
		for n in kwargs:
			self._blobs[n] = 0*kwargs[n]
		
		for n,l,i,o in self._layers:
			for s in i+o:
				if not s in self._blobs:
					self._blobs[s] = np.empty((0,))
			l.setup( self._get_blobs(i), self._get_blobs(o) )
		for n in self._blobs:
			self._diffs[n] = 0*self._blobs[n]
	
	def forward1(self,**kwargs):
		"""
		    Run a single forward pass (not recurrently)
		    Initialize the blob values using the keyword arguments
		"""
		for n in kwargs:
			self._blobs[n][...] = kwargs[n]
		
		for n,l,i,o in self._layers:
			l.forward( self._get_blobs(i), self._get_blobs(o) )
	
	def backward1(self,**kwargs):
		"""
		    Run a single backward pass (not recurrently)
		    Initialize the diff values using the keyword arguments
		"""
		for n in kwargs:
			self._diffs[n][...] = kwargs[n]
		
		for n,l,i,o in self._layers[::-1]:
			l.backward( self._get_blobs(i), self._get_blobs(o), self._get_diffs(i), self._get_diffs(o) )
	
	def forwardMany(self,N=None,**kwargs):
		"""
		    Call forward for a sequence of length (N). N can also
		    be inferred from kwargs if any are given.
		    Note: this does not store the indidual network states,
		    but rather overwrites them.
		"""
		# Set all the blobs to 0
		for b in self._blobs.values():
			b[...] = 0
		
		if N == None: N = kwargs.values()[0].shape[0]
		
		# Run forward
		for it in range(N):
			self.forward1( {n:kwargs[n][i] for n in kwargs} )
		
		
	def forwardBackwardAll(self,N=None,**kwargs):
		"""
		    Call forward and backward for a sequence of length (N).
		    N can also be inferred from kwargs if any are given.
		    Returns the flat gradient and the loss for each iterations
		"""
		# Get the set of blobs that have a loss
		blob_loss_weight = {s:l.loss_weight for n,l,i,o in self._layers for s in o if l.loss_weight != 0}
		print( blob_loss_weight )
		# Set all the blobs to 0
		for b in self._blobs.values():
			b[...] = 0
		
		if N == None:
			assert len(kwargs.values()) > 0
			N = kwargs[list(kwargs)[0]].shape[0]
		
		# Run forward
		for it in range(N):
			self.forward1( **{n:kwargs[n][it] for n in kwargs} )
			self.pushState()
		
		# Set the loss diffs to 0
		for b in self._diffs.values():
			b[...] = 0
		
		for n in blob_loss_weight:
			self._diffs[n][...] = blob_loss_weight[n]
		
		# Run backward and accumulate the flat gradient
		g = 0*self.flat_gradient
		losses = []
		for it in range(N):
			self.popState()
			self.backward1( **{n:kwargs[n][it] for n in kwargs} )
			g += self.flat_gradient
			losses.append( {n:np.sum(self._blobs[n])*blob_loss_weight[n] for n in blob_loss_weight} )
		return g, losses
	
	@property
	def blobs(self):
		return self._blobs
	
	@property
	def diffs(self):
		return self._diffs
	
	@property
	def gradient(self):
		"""
		    Return the gradient of all layers
		"""
		r = {}
		for n,l,i,o in self._layers:
			if not n in r:
				r[n] = l.gradient
			else:
				# Add the parmaters
				g = l.gradient
				assert set(g.keys()) == set(m.keys())
				for m in g:
					r[n][m] += g[m]
		return r
	
	@property
	def flat_gradient(self):
		"""
		    Return the gradient of all layers
		"""
		r = {}
		for n,l,i,o in self._layers:
			if not n in r:
				r[n] = l.flat_gradient
			else:
				# Add the parmaters
				g = l.flat_gradient
				assert set(g.keys()) == set(m.keys())
				for m in g:
					r[n][m] += g[m]
		return np.concatenate( [r[n] for n in sorted(r.keys())], axis=0 )
	
	def pushState(self):
		"""
		    Push the current blobs on a stack (so that we can remember them later)
		"""
		import copy
		self._blob_history.append( copy.deepcopy(self._blobs) )
	
	def popState(self):
		self._blobs = self._blob_history.pop()
	
	@property
	def parameters(self):
		""" Return the parameters of the model """
		r = {}
		for n,l,i,o in self._layers:
			if not n in r:
				r[n] = l.parameters
			else:
				# Add the parmaters
				g = l.parameters
				assert set(g.keys()) == set(m.keys())
				for m in g:
					r[n][m] += g[m]
		return r
	
	@property
	def flat_parameters(self):
		""" Return the flat parameters of the model """
		r = {}
		for n,l,i,o in self._layers:
			if not n in r:
				r[n] = l.flat_parameters
			else:
				# Add the parmaters
				g = l.flat_parameters
				assert set(g.keys()) == set(m.keys())
				for m in g:
					r[n][m] += g[m]
		return np.concatenate( [r[n] for n in sorted(r.keys())], axis=0 )
	@flat_parameters.setter
	def flat_parameters(self, value):
		""" TODO: Implement """
		pass
