import network
from layers import *

l = network.Network()
l.addLayer( 'ReLU1', ReLU(), 'data', 'positive_data' )
l.addLayer( 'InnerProduct1', InnerProduct(output_size=10), 'data', 'score' )
l.addLayer( 'Sigm', Sigmoid(), 'score', 'out' )

l.setup( data=np.ones((5,10)) )
print( [(n,v.shape) for n,v in l.blobs.items()] )

l.forward1( data=np.random.normal(0,10,(5,10)).astype(int) )
l.backward1( out=np.random.random(l.blobs['out'].shape)>0.5 )
print( l.forwardBackwardAll( data = np.random.normal(0,10,(20,5,10)) ) )
#print( l.blobs['data'] )
#print( l.blobs['positive_data'] )
#print( l.blobs['out'] )
#print( l.diffs['data'] )
print( l.parameters )
print( l.flat_parameters )

