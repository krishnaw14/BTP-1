import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VQVAETrainer(object):

	def __init__(self, model, config):
		self.model = model
		self.optimizer = self.get_optimizer(config['train']['optim'], config['train']['lr'])
		self.train_loader = self.get_data_loader(config['data']['name'], config['data']['path'], config['train']['batch_size'], train=True)
		self.val_loader = self.get_data_loader(config['data']['name'], config['data']['path'], config['val']['batch_size'], train=False)
		self.num_epochs = config['train']['num_epochs']
		self.val_step = config['val']['interval_step']

		self.param_save_dir = config['model']['save_dir']
		self.save_param_step = config['model']['save_interval']
		self.log_result_step = config['train']['log_interval']
		self.log_result_dir = config['train']['log_result_dir']
		
		self.load_checkpoint(config['model']['checkpoint'])

		os.makedirs(self.param_save_dir, exist_ok=True)
		os.makedirs(self.log_result_dir, exist_ok=True)

		# self.siamese_step
		self.siamese_coefficient = config['siamese']['coefficient']

	
	def get_optimizer(self, optim_name, lr):
		if optim_name == 'adam':
			return torch.optim.Adam(self.model.parameters(), lr=lr)

	def load_dataset(self, dataset_name, dataset_path):
		if dataset_name == 'mnist':
			self.train_dataset = datasets.MNIST(root=dataset_path, train=True, download=True,
				transform=transforms.Compose([
					transforms.ToTensor()
					# transforms.Normalize((0.5*255.0, 0.5*255.0, 0.5*255.0 ), (1.0*255.0, 1.0*255.0, 1.0*255.0 ))
					]))
			self.val_dataset = datasets.MNIST(root=dataset_path, train=False, download=True,
				transform=transforms.Compose([
					transforms.ToTensor()
					# transforms.Normalize((0.5*255.0, 0.5*255.0, 0.5*255.0 ), (1.0*255.0, 1.0*255.0, 1.0*255.0 ))
					]))

			self.data_variance = np.var(self.train_dataset.data.numpy() / 255.0)

	def get_data_loader(self, dataset_name, dataset_path, batch_size, train=True):
		if train == True:
			self.load_dataset(dataset_name, dataset_path)
			return DataLoader(self.train_dataset, batch_size=batch_size,
				shuffle=True, pin_memory=True)
		else:
			return DataLoader(self.val_dataset, batch_size=batch_size,
				shuffle=True, pin_memory=True)

	def save_checkpoint(self, epoch, total_loss, recon_loss, is_best=False):
		torch.save({
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'total_loss': total_loss, 'recon_loss': recon_loss,
			'is_best': is_best
			}, os.path.join(self.param_save_dir, 'epoch_{}.tar'.format(epoch))
			)

	def load_checkpoint(self, checkpoint_path):
		if os.path.exists(checkpoint_path):
			checkpoint = torch.load(checkpoint_path)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	def forward_pass(self, img, label):
		def compute_contrastive_loss(x1,x2,y,margin=2):
			output1, output2 = self.model.siamese(x1,x2)
			# import pdb; pdb.set_trace()
			euclidean_distance = F.pairwise_distance(output1, output2)
			loss_contrastive = torch.mean((1-y.float()) * torch.pow(euclidean_distance, 2) +
                                      (y.float()) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
			return loss_contrastive

		latent_loss, img_recon, q_z_e, perplexity = self.model(img)
		recon_loss = torch.mean((img_recon-img)**2)/self.data_variance
		total_loss = recon_loss + latent_loss

		siamese_loss = 0
		for i in range(self.train_loader.batch_size-1):
			siamese_loss += compute_contrastive_loss(q_z_e[i], q_z_e[i+1], label[i] == label[i-1])

		total_loss = recon_loss + latent_loss + self.siamese_coefficient*siamese_loss

		return total_loss, recon_loss, img_recon, siamese_loss

	def save_results(self, epoch, img_recon):
		# concat = torch.cat((img[:16], img_recon[:16]))
		save_image(img_recon, os.path.join(self.log_result_dir, 'epoch_{}.png'.format(epoch)), nrow=8, normalize=True)

	def train(self):
		for epoch in range(self.num_epochs):
			pbar = tqdm(enumerate(self.train_loader), desc = 'training batch_loss')
			epoch_loss = 0.0
			for (nbatch, data) in pbar:
				img, label = data
				img = img.to(device)
				self.optimizer.zero_grad()

				total_loss, recon_loss, img_recon, siamese_loss = self.forward_pass(img, label)
				if epoch%self.log_result_step == 0 and nbatch == 0:
					self.save_results(epoch, img_recon)
				total_loss.backward()

				self.optimizer.step()

				pbar.set_description('Epoch: {}, Batch Loss: {}, Recon Loss {}, Siamese Loss: {}'.format(epoch, total_loss, recon_loss, siamese_loss))
				epoch_loss += total_loss
			avg_epoch_loss = epoch_loss/(self.train_loader.dataset.data.shape[0]//self.train_loader.batch_size)

			print('Average Epoch Loss:', avg_epoch_loss)

			if epoch%self.val_step == 0:
				self.validate()
			if epoch%self.save_param_step == 0:
				self.save_checkpoint(epoch, total_loss, recon_loss)
			# if epoch%self.siamese_step == 0:
			# 	self.


	def validate(self):
		pass