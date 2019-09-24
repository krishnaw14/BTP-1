import torch 
import torch.nn as nn 
import torch.nn.init as init

def weight_init(m, init_mode='normal_init'):
	if isinstance(m, (nn.Linear, nn.Conv2d)):
		if init_mode == 'kaiming_init':
			init.kaiming_normal_(m.weight)
		else:
			init.normal_(m.weight, 0, 0.02)

		if m.bias is not None:
			m.bias.data.fill_(0)

	elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
		m.weight.data.fill_(1)
		if m.bias is not None:
			m.bias.data.fill_(0)

class Discriminator(nn.Module):
	def __init__(self, config):
		super(Discriminator, self).__init__()

		self.z_dim = config['model']['z_dim']
		self.hidden_dim = config['model']['discriminator']['hidden_dim']
		self.out_dim = config['model']['discriminator']['out_dim']
		self.num_layers = config['model']['discriminator']['num_layers'] # 5
		self.bn = config['model']['discriminator']['bn']
		self.module_list = nn.ModuleList()
		for i in range(self.num_layers):
			if i == 0:
				self.module_list.append(nn.Linear(self.z_dim, self.hidden_dim))
			elif i == self.num_layers-1:
				self.module_list.append(nn.Linear(self.hidden_dim, self.out_dim))
			else:
				self.module_list.append(nn.Linear(self.hidden_dim, self.hidden_dim))

			if self.bn[i]:
				if i != self.num_layers-1:
					self.module_list.append(nn.BatchNorm1d(self.hidden_dim))
				else:
					self.module_list.append(nn.BatchNorm1d(self.out_dim))
			if i != self.num_layers-1:
				self.module_list.append(nn.LeakyReLU(0.2, True)) 

		self.net = nn.Sequential(*self.module_list)

		self.init_mode = config['model']['init_mode']
		self.weight_init()

	def weight_init(self):
		for block in self._modules:
			for m in self._modules[block]:
				weight_init(m, init_mode=self.init_mode)

	def forward(self, z):
		out = self.net(z).squeeze()
		return out

class FactorVAE(nn.Module):

	def __init__(self, config, device):
		super(FactorVAE, self).__init__()

		self.z_dim = config['model']['z_dim']
		self.img_channels = config['model']['img_channels']

		self.encoder_num_layers = config['model']['encoder']['num_layers']
		self.encoder_channels = config['model']['encoder']['channels']
		self.encoder_kernel_sizes = config['model']['encoder']['kernel_sizes']
		self.encoder_strides = config['model']['encoder']['strides']
		self.encoder_padding = config['model']['encoder']['padding']
		self.encoder_bn = config['model']['encoder']['batch_norm']

		self.decoder_num_layers = config['model']['decoder']['num_layers']
		self.decoder_channels = config['model']['decoder']['channels']
		self.decoder_kernel_sizes = config['model']['decoder']['kernel_sizes']
		self.decoder_strides = config['model']['decoder']['strides']
		self.decoder_padding = config['model']['decoder']['padding']
		self.decoder_bn = config['model']['decoder']['batch_norm']

		encoder_modules = nn.ModuleList()
		for i in range(self.encoder_num_layers):
			in_channels = self.img_channels if i == 0 else self.encoder_channels[i-1]
			out_channels = self.encoder_channels[i]
			kernel_size = self.encoder_kernel_sizes[i]
			stride = self.encoder_strides[i]
			padding = self.encoder_padding[i]

			encoder_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
			if self.encoder_bn[i]:
				encoder_modules.append(nn.BatchNorm2d(out_channels))
			if i != self.encoder_num_layers-1:
				encoder_modules.append(nn.ReLU())

		decoder_modules = nn.ModuleList()
		for i in range(self.decoder_num_layers):
			in_channels = self.z_dim if i == 0 else self.decoder_channels[i-1]
			out_channels = self.img_channels if i == self.decoder_num_layers-1 else self.decoder_channels[i]
			kernel_size = self.decoder_kernel_sizes[i]
			stride = self.decoder_strides[i]
			padding = self.decoder_padding[i]

			if i==0:
				decoder_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
			else:
				decoder_modules.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
			if self.decoder_bn[i]:
				decoder_modules.append(nn.BatchNorm2d(out_channels))
			if i != self.decoder_num_layers-1:
				decoder_modules.append(nn.ReLU(inplace=True))

		self.encoder = nn.Sequential(*encoder_modules)
		self.decoder = nn.Sequential(*decoder_modules)

		self.init_mode = config['model']['init_mode']
		self.weight_init()

	def weight_init(self):
		for block in self._modules:
			for m in self._modules[block]:
				weight_init(m, init_mode=self.init_mode)	

	def encode(self, x):
		h = self.encoder(x)
		h = h.view(z.size(0), -1)
		mu, logvar = self.fc_mu(h), self.fc_logvar(h)
		return mu, logvar

	def decode(self, z):
		h = self.fc_decoder(z)
		h = h.view(h.size(0), -1, 1, 1)
		xrecon = self.decoder(h)
		return xrecon

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = std.data.new(std.size()).normal_()
		z = mu + std*eps
		return z

	def forward(self, x, no_decoder=False):
		encoder_out = self.encoder(x)
		mu, logvar = encoder_out[:, :self.z_dim], encoder_out[:,self.z_dim:]
		z = self.reparameterize(mu, logvar)
		if no_decoder:
			return z.squeeze()
		xrecon = self.decoder(z)

		return xrecon, mu, logvar, z.squeeze()
