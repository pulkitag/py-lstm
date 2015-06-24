from layers import *
import numpy as np
import scipy.optimize as sopt
class GradChecker:
	def __init__(self):
		self.layer_ = []
		self.bot_ = []
		self.top_ = []
		self.w_ = []
	
	def setup(self, layer, bot, epsilon=0.001):
		self.layer_ = layer
		self.bot_ = bot
		self.top_ = np.empty((0,))
		layer.setup([self.bot_], [self.top_])
		self.w_ = epsilon * np.random.rand(self.top_.shape[0])
	
	def gradient(self, bot):
		bot = bot[0]
		top = np.zeros_like(self.top_)
		self.layer_.forward([bot], [top])
		botgrad = 0 * bot
		topgrad = self.w_
		self.layer_.backward([bot], [top], [botgrad], [topgrad])
		return np.sum(botgrad)
	
	def forward(self, bot):
		bot = bot[0]
		top = np.zeros_like(self.top_)
		self.layer_.forward([bot], [top])
		return np.sum(self.w_ * top)

	def check(self, bot):
		res = sopt.check_grad(self.forward, self.gradient, [bot])
		return res

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Gradient checking for a layer')
	parser.add_argument('layer', type=str, help='Layer name')
	parser.add_argument('D', type=int, help='#input dimensions')
	parser.add_argument('runs', type=int, help='#random simulations')
	args = parser.parse_args()
	layer = ReLU()
	res = 0
	for n in range(args.runs):
		layer = eval(args.layer)
		bot = (np.random.rand(args.D) - 0.5)
		if not isinstance(layer, BaseLayer):
			layer = layer()
		gc = GradChecker()
		gc.setup(layer, bot)
		res += gc.check(bot)
	print('Mean residual: ', res/args.runs)