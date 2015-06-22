import network
from layers import *

l = network.Network()
l.addLayer( 'ReLU1', ReLU(), 'data', 'positive_data' )

l.setup( data=np.ones((10,10)) )
