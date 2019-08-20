import nnabla as nn
import nnabla.functions as F
import nnabla.inializer as I
import nnabla.parametric_functions as PF
import numpy as np

class ActNorm(object):

	def __init__(self, in_channel, logdet=True):
		self.loc = nn.get_parameter_or_create
		self.scale = nn.get_parameter_or_create

		self.logdet = False

	def ddi(self, x):

	def forward(self, x):
		N, C, H, W = x.shape

		log_abs = F.log(F.abs(self.scale))
		logdet = H*W*F.sum(log_abs)

		if self.logdet:
			return self.scale * (x + self.loc), logdet
		else:
			return self.scale * (x + self.loc)

	def reverse(self, x):
		out = x/self.scale - self.loc

class InvConv(object):

	def __init__(self, in_channel, param_name):
		w_init = np.linalg.qr(np.random.randn(in_channel,in_channel))[0]
		self.weight = nn.get_parameter_or_create(param_name, shape=w_init.shape, 
			initializer=w_init)

	def forward(self, x):
		N, C, H, W = x.shape
		out = F.conv2d(input, self.weight.reshape(self.weight.shape+(1,1)))
		dlogdet = H*W*F.log(np.abs(np.linalg.det(self.weight)))
        return out, logdet

	def reverse(self, x):
		weight_inv = nn.Variable(self.weight.shape)
		weight_inv.d = np.linalg.inv(self.weight.d)
		out = F.convolution(x, weight_inv)
		return out

class AffineCoupling(object):

	def __init__(self, in_channel, filter_size=512, affine=True, scope_name='affine_coupling'):

		self.affine = affine
		self.in_channel = in_channel
		self.filter_size = filter_size
		self.scale = nn.get_parameter_or_create(scope_name+'/scale', shape=(1,self.in_channel,1,1),
			initializer=)

	def conv_block(self, x):
		out = PF.convolution(x_a, self.filter_size, (3,3), pad=(1,1))
		out = F.relu(out, inplace=True)
		out = PF.convolution(out, self.filter_size, (1,1))
		out = F.relu(out, inplace=True)

		out = F.pad
		out = PF.convolution(out, self.in_channel, (3,3), pad=(0,0))
		out = out*F.exp(self.scale*3)

		return out

	def forward(self, x, scope_name):
		x_a, x_b = 
		if self.affine:
			with nn.paramter_scope(scope_name):
				out = self.conv_block(x_a)

	def backward(self, x, scope_name):
		x_a, x_b = 
		if self.affine:
			out = self.conv_block(x_a)

class Flow(object):
	def __init__(self, conv_lu=False):
		self.actnorm = ActNorm()
		if conv_lu:
			self.invconv = InvConvLU()
		else:
			self.invconv = InvConv()
		self.affine_coupling = AffineCoupling()

	def forward(self, x):
		out, logdet = self.actnorm(x)
		out, det1 = self.invconv(out)
		out, det2 = self.affine_coupling(out)

		logdet += det1
		if det2 is not None:
			logdet += det2

		return out, logdet

	def reverse(self, x):
		out = self.affine_coupling.reverse(x)
		out = self.invconv.reverse(out)
		out = self.actnorm.reverse(out)

		return out

class Block(object):

	def __init__(self, in_channels, n_flow, split=True, affine=True, conv_lu=True):

		squeeze_dim = in_channel*4
		# define prior in case of 

	def flow(self, x):

	def forward():

	def reverse():

	def __call__(self)

class Glow(object):

	def __init__(self, config):

		self.in_channels = config['model']['in_channels']
		self.affine = config['model']['affine']
		self.conv_lu = config['model']['conv_lu']
		self.n_blocks = config['model']['n_blocks']
		self.n_flow = config['model']['n_flow']

	# def block(self, x, in_channels, split=True, affine=True, conv_lu=True, 
	# 	scope_name='default_scope_name'):
		
	# 	squeeze_dim = in_channel*4
	# 	# Define Prior in case of split
	# 	N,C,H,W = x.size
	# 	squeezed_x = x.reshape((N, C, H // 2, 2, W//2, 2))
	# 	squeezed_x = F.transpose(squeezed_x, (0, 1, 3, 5, 2, 4))
	# 	out = squeezed_x.reshape((N,C*4, H//2, W//2))
	# 	logdet = 0

	# 	with nn.parameter_scope(scope):
	# 		out, det = flow()


	def forward(self, x):

	def reverse(self, x):

	def __call__(self, x):
