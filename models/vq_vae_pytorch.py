import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
	def __init__(self, channels):
		super(ResidualBlock, self).__init__()
		self.res_block = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(channels),
			nn.ReLU(),
			nn.Conv2d(channels, channels, kernel_size=1, stride=1),
			nn.BatchNorm2d(channels)
			)

	def forward(self, x):
		y = x + self.res_block(x)
		return y

class SiameseNetwork(nn.Module):
	def __init__(self, channels, dim):
		super(SiameseNetwork, self).__init__()
		self.conv_layer = nn.Sequential(
			ResidualBlock(channels),
			nn.ReLU(),
			ResidualBlock(channels),
			nn.Conv2d(channels, channels*2, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(channels*2),
			nn.ReLU(),
			nn.Conv2d(channels*2,channels*2, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(channels*2),
			nn.ReLU(),
			)
		self.fc_layer = nn.Sequential(
			nn.Linear(channels*2*2*2, dim),
			nn.ReLU(),
			nn.Linear(dim, dim),
			nn.ReLU()
			)
	def forward(self, x1, x2):
		# import pdb; pdb.set_trace()
		out1, out2 = self.conv_layer(x1.unsqueeze(0)), self.conv_layer(x2.unsqueeze(0))
		out1, out2 = self.fc_layer(out1.view(1, -1)), self.fc_layer(out2.view(1, -1))
		return out1, out2


class VectorQuantizer(nn.Module):
	def __init__(self, K, D, beta):
		super(VectorQuantizer, self).__init__()
		self.K = K
		self.D = D
		self.beta = beta

		self.embedding = nn.Embedding(self.K, self.D)
		self.embedding.weight.data.uniform_(-1/K, 1/K)

	def forward(self, z_e):

		z_e = z_e.permute(0,2,3,1).contiguous()
		z_e_flat = z_e.view(-1, self.D)

		distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) + \
					torch.sum(self.embedding.weight**2, dim=1) - \
					2*torch.matmul(z_e_flat, self.embedding.weight.t())
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

		encodings = torch.zeros(encoding_indices.shape[0], self.K).to(device)
		encodings.scatter_(1, encoding_indices, 1)

		quantized = self.embedding(encoding_indices).view(z_e.shape)

		e_latent_loss = torch.mean((quantized.detach() - z_e)**2)
		q_latent_loss = torch.mean((quantized - z_e.detach())**2)
		loss = q_latent_loss + self.beta*e_latent_loss

		quantized = z_e + (quantized-z_e).detach()
		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs*torch.log(avg_probs+1e-10)))

		return quantized.permute(0,3,1,2).contiguous(), loss, perplexity, encodings


class VQVAE(nn.Module):
	def __init__(self, config):
		super(VQVAE, self).__init__()
		encoder_hidden_dim = config['encoder']['hidden_dim']
		decoder_hidden_dim = config['decoder']['hidden_dim']
		self.encoder = nn.Sequential(
			nn.Conv2d(
			in_channels=config['encoder']['in_channels'],
			out_channels=encoder_hidden_dim,
			kernel_size=4,
			stride=2,
			padding=1
			),
			nn.BatchNorm2d(encoder_hidden_dim),
			nn.ReLU(inplace=True),

			nn.Conv2d(
			in_channels=encoder_hidden_dim,
			out_channels=encoder_hidden_dim,
			kernel_size=4,
			stride=2,
			padding=1
			),
			nn.BatchNorm2d(encoder_hidden_dim),
			nn.ReLU(inplace=True),

			ResidualBlock(encoder_hidden_dim),
			nn.BatchNorm2d(encoder_hidden_dim),
			ResidualBlock(encoder_hidden_dim),
			nn.BatchNorm2d(encoder_hidden_dim)
			)

		self.decoder = nn.Sequential(
			ResidualBlock(decoder_hidden_dim),
			nn.BatchNorm2d(decoder_hidden_dim),
			ResidualBlock(decoder_hidden_dim),
			nn.ConvTranspose2d(
				decoder_hidden_dim, decoder_hidden_dim,
				kernel_size=4, stride=2, padding=1
				),
			nn.BatchNorm2d(decoder_hidden_dim),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(decoder_hidden_dim, config['decoder']['out_channels'],
				kernel_size=4, stride=2, padding=1)
			)

		self.vector_quantizer = VectorQuantizer(
			config['vector_quantizer']['num_embeddings'],
			config['vector_quantizer']['embedding_dim'],
			config['vector_quantizer']['commitment_cost']
			)

		self.siamese = SiameseNetwork(
			config['siamese']['channels'],
			config['siamese']['dim'])

	def forward(self, x):

		z_e = self.encoder(x)
		q_z_e, loss, perplexity, encodings = self.vector_quantizer(z_e)
		# import pdb; pdb.set_trace()
		x_recon = self.decoder(q_z_e)

		return loss, x_recon, q_z_e, perplexity




