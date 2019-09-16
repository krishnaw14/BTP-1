import torch 
import torch.nn as nn 

class VAE(nn.Module):

	def __init__(self, config, device):

		encoder_modules = nn.ModuleList()
		for i in config['model']['encoder']['num_conv_layers']:
			in_channels = config['model']['encoder']['in_channels'] if i == 0 else config['model']['encoder']['channels'][i-1]
			out_channels = config['model']['encoder']['channels'][i]
			kernel_size = config['model']['encoder']['kernel_sizes'][i]
			stride = config['model']['encoder']['strides'][i]
			padding = config['model']['encoder']['padding'][i]

			encoder_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
			if config['model']['encoder']['batch_norm'][i]:
				encoder_modules.append(nn.BatchNorm2d(out_channels))
			encoder_modules.append(self.get_activation_function_by_name(config['model']['encoder']['activation']))

		self.fc_mu = nn.Linear(config['model']['h_dim'], config['model']['z_dim'])
		self.fc_logvar= nn.Linear(config['model']['h_dim'], config['model']['z_dim'])

		decoder_modules = nn.ModuleList()
		for i in config['model']['decoder']['num_conv_layers']:
			in_channels = config['model']['decoder']['channels'][-1] if i == 0 else config['model']['decoder']['channels'][i-1]
			out_channels = config['model']['decoder']['channels'][i] 
			kernel_size = config['model']['decoder']['kernel_sizes'][i]
			stride = config['model']['decoder']['strides'][i]
			padding = config['model']['decoder']['padding'][i]

			decoder_modules.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
			if config['model']['decoder']['batch_norm'][i]:
				decoder_modules.append(nn.BatchNorm2d(out_channels))
			decoder_modules.append(self.get_activation_function_by_name(config['model']['decoder']['activation']))

		self.fc_decoder = nn.Linear(config['model']['z_dim'], config['model']['h_dim'])

		self.encoder = nn.Sequential(*encoder_modules)
		self.decoder = nn.Sequential(*decoder_modules)

	def get_activation_function_by_name(self, activation_funtion_name):
		if activation_funtion_name == 'relu':
			return nn.ReLU(inplace=True)
		elif activation_funtion_name == 'tanh':
			return nn.Tanh()
		elif activation_funtion_name == ''

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
		eps = torch.randn(*mu.size())
		z = mu + std*esp
		return z

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		xrecon = self.decode(z)

		return xrecon, z, mu, logvar
