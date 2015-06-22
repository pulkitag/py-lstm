

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
	
	def addLayer(self,layer_instance,input_blobs,output_blobs):
		"""
		    Add a new layer_instance with inputs and outputs given as a list of strings
		"""
		if not isinstance(input_blobs ,list): input_blobs  = [input_blobs]
		if not isinstance(output_blobs,list): output_blobs = [output_blobs]
		self.layers.append( (layer_instance,input_blobs,output_blobs) )
	
	
	def setup(self,**kwargs):
		"""
		    Setup the network
		"""
		for l,i,o in self.layers:
			for n in i+o:
				if not n in self.blobs:
					self.blobs[n] = np.empty()
			
		
	
	def forward1(self,**kwargs):
		"""
		    Run a single forward pass (not recurrently)
		    Initialize the blob values using the keyword arguments
		"""
		for n,v in kwargs:
			self.blobs[n][...] = v
		
		for l,i,o in self.layers:
			l.forward( self._get_blobs(i), self._get_blobs(o) )
	
	def backward1(self,**kwargs):
		"""
		    Run a single backward pass (not recurrently)
		    Initialize the diff values using the keyword arguments
		"""
		for n,v in kwargs:
			self.diffs[n][...] = v
		
		for l,i,o in self.layers[::-1]:
			l.backward( self._get_blobs(i), self._get_blobs(o) )
		
	#def forwardBackward(self,**kwargs):
		#pass
	def gradient(self):
		"""
		    Push the current blobs on a stack (so that we can remember them later)
		"""
		pass
	
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
		""" Fetch all the parameters of the network """
		pass
	@parameters.setter
	def x(self, value):
		""" Set all the parameters of the network """
		pass
