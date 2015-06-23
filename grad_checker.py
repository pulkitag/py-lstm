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
		self.w_ = epsilon * np.random.rand(self.top_.shape)
	
	def gradient(self, bot):
		top = np.zeros_like(self.top_)
		self.layer_.forward([bot], [top])
		botgrad = 0 * bot
		topgrad = self.w_
		self.layer_.backward([bot], [top], [botgrad], [topgrad])
		return botgrad
	
	def forward(self, bot):
		top = np.zeros_like(self.top_)
		self.layer_.forward([bot], [top])
		return self.w_ * top

	def check(self, bot):
		res = sopt.check_grad(forward(bot), gradient(bot), bot)


if __name__ == "__main__":
	def main():



