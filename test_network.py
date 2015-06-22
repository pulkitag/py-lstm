import network
from layers import *

l = network.Network()
l.addLayer( 'ReLU1', ReLU(), 'data', 'positive_data' )


