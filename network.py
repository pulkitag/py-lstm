

class Network:
	def __init__(self):
		# Stores layers in the format (layer_object,input_names,output_names)
		self.layers = []
		self.blobs = {}
		self.diffs = {}
		self.blob_history = []
	
	def _get_blobs(self,names):
		return [self.blobs[n] for n in names]
	
	def _get_diffs(self,names):
		return [self.diffs[n] for n in names]
	
	def addLayer(self,layer_name,layer_instance,input_blobs,output_blobs):
		"""
		    Add a new layer_instance with inputs and outputs given as a list of strings
		"""
		if not isinstance(input_blobs ,list): input_blobs  = [input_blobs]
		if not isinstance(output_blobs,list): output_blobs = [output_blobs]
		self.layers.append( (layer_name,layer_instance,input_blobs,output_blobs) )
	
	def setup(self,**kwargs):
		"""
		    Setup the network
		"""
		for n,l,i,o in self.layers:
			for s in i+o:
				if not s in self.blobs:
					self.blobs[s] = np.empty()
			l.setup( self._get_blobs(i), self._get_blobs(o) )
	
	def forward1(self,**kwargs):
		"""
		    Run a single forward pass (not recurrently)
		    Initialize the blob values using the keyword arguments
		"""
		for n,v in kwargs:
			self.blobs[n][...] = v
		
		for n,l,i,o in self.layers:
			l.forward( self._get_blobs(i), self._get_blobs(o) )
	
	def backward1(self,**kwargs):
		"""
		    Run a single backward pass (not recurrently)
		    Initialize the diff values using the keyword arguments
		"""
		for n,v in kwargs:
			self.diffs[n][...] = v
		
		for n,l,i,o in self.layers[::-1]:
			l.backward( self._get_blobs(i), self._get_blobs(o), self._get_diffs(i), self._get_diffs(o) )
		
	#def forwardBackward(self,**kwargs):
		#pass
	@property
	def gradient(self):
		"""
		    Return the gradient of all layers
		"""
		r = {}
		for n,l,i,o in self.layers:
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
		for n,l,i,o in self.layers:
			if not n in r:
				r[n] = l.flat_gradient
			else:
				# Add the parmaters
				g = l.flat_gradient
				assert set(g.keys()) == set(m.keys())
				for m in g:
					r[n][m] += g[m]
		return np.concatenate( r.values(), axis=0 )
	
	def pushState(self):
		"""
		    Push the current blobs on a stack (so that we can remember them later)
		"""
		import copy
		self.blob_history.append( copy.deepcopy(self.blobs) )
	
	def popState(self):
		self.blobs = self.blob_history.pop()
	
	@property
	def parameters(self):
		""" TODO: Implement """
		pass
	@property
	def flat_parameters(self, value):
		""" TODO: Implement """
		pass
	@flat_parameters.setter
	def flat_parameters(self, value):
		""" TODO: Implement """
		pass
