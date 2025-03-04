from .base import Trainer
from misc.factor_vae import kl_divergence, permute_dims, get_data

from tqdm import tqdm
import os
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision.utils import save_image

class FactorVAETrainer(Trainer):

	def __init__(self, vae_model, discriminator, config, device):
		self.vae_model = vae_model
		self.discriminator = discriminator
		self.vae_optimizer = self.get_optimizer(self.vae_model.parameters(), config['train']['optim'], 
			config['train']['lr_vae'])
		self.discriminator_optimizer = self.get_optimizer(self.discriminator.parameters(), config['train']['optim'], 
			config['train']['lr_discriminator'], betas=(0.5,0.9))
		self.train_loader = self.get_data_loader(config['data'], train=True)
		# self.val_loader = self.get_data_loader(config['data'], train=False)
		self.batch_size = self.train_loader.batch_size
		self.num_epochs = config['train']['num_epochs']
		self.val_step = config['val']['interval_step']

		self.param_save_dir = config['model']['save_dir']
		self.save_param_step = config['model']['save_interval']
		self.log_result_step = config['train']['log_interval']
		self.log_result_dir = config['train']['log_result_dir']

		self.gamma = config['model']['gamma']

		self.device = device
		
		self.load_checkpoint(config['model']['checkpoint'])

		os.makedirs(self.param_save_dir, exist_ok=True)
		os.makedirs(self.log_result_dir, exist_ok=True)

		self.ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
		self.zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
		self.iters_per_epoch = np.ceil(len(self.train_loader.dataset)/self.train_loader.batch_size)

	def get_data_loader(self, data_config, train=True):
		if train:
			self.train_dataset, self.val_dataset = get_data(data_config)
			return DataLoader(self.train_dataset, batch_size=data_config['batch_size'],
				shuffle=True, pin_memory=True)
		else:
			return DataLoader(self.val_dataset, batch_size=data_config['batch_size'],
				shuffle=True, pin_memory=True)

	def save_checkpoint(self, epoch, vae_loss, discriminator_loss, is_best=False):
		torch.save({
			'vae_model_state_dict': self.vae_model.state_dict(),
			'discriminator_state_dict': self.discriminator.state_dict(),
			'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
			'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
			'vae_loss': vae_loss, 'discriminator_loss': discriminator_loss,
			'is_best': is_best
			}, os.path.join(self.param_save_dir, 'epoch_{}.tar'.format(epoch))
			)

	def load_checkpoint(self, checkpoint_path):
		if os.path.exists(checkpoint_path):
			checkpoint = torch.load(checkpoint_path)
			self.vae_model.load_state_dict(checkpoint['model_state_dict'])
			self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	def vae_forward_pass(self, img):
		img_recon, mu, logvar, z = self.vae_model(img)
		recon_loss = F.binary_cross_entropy_with_logits(img_recon, img)
		kld = kl_divergence(mu, logvar)

		D_z = self.discriminator(z)
		tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

		total_vae_loss = recon_loss + kld + self.gamma*tc_loss

		return total_vae_loss, recon_loss, z

	def discriminator_forward_pass(self, img_2, img_1):

		z_prime = self.vae_model(img_2, no_decoder=True)
		z_perm = permute_dims(z_prime).detach()
		D_z_perm = self.discriminator(z_perm)

		z = self.vae_model(img_1, no_decoder=True)
		D_z = self.discriminator(z)

		discriminator_loss = 0.5*(F.cross_entropy(D_z, self.zeros) + F.cross_entropy(D_z_perm, self.ones))
		return discriminator_loss

	def train(self):
		# import pdb; pdb.set_trace()
		for epoch in range(self.num_epochs):
			pbar = tqdm(enumerate(self.train_loader), desc = 'training batch_loss', total=self.iters_per_epoch)
			epoch_vae_loss = 0.0
			epoch_discriminator_loss = 0.0 
			for (nbatch, data) in pbar:
				img_1, img_2 = data
				img_1, img_2 = img_1.to(self.device), img_2.to(self.device)

				total_vae_loss, recon_loss, z = self.vae_forward_pass(img_1)
				# if epoch%self.log_result_step == 0 and nbatch == 0:
				# 	self.save_results(epoch, img_recon)
				self.vae_optimizer.zero_grad()
				total_vae_loss.backward(retain_graph=False)
				self.vae_optimizer.step()

				discriminator_loss = self.discriminator_forward_pass(img_2, img_1)
				self.discriminator_optimizer.zero_grad()
				discriminator_loss.backward(retain_graph=False)
				self.discriminator_optimizer.step()

				epoch_vae_loss += total_vae_loss
				epoch_discriminator_loss += discriminator_loss

				# pbar.set_description('Epoch: {}, VAE Loss: {}, Discriminator Loss {}, Recon Error: {}'.format(epoch, 
				# 	total_vae_loss, discriminator_loss, recon_loss))
				
			avg_epoch_vae_loss = epoch_vae_loss/(self.iters_per_epoch)
			avg_epoch_discriminator_loss = epoch_discriminator_loss/(self.iters_per_epoch)

			print('Average VAE Loss:', avg_epoch_vae_loss)
			print('Average Discriminator Loss:', avg_epoch_discriminator_loss)

			if epoch%self.val_step == 0:
				self.validate()
			if epoch%self.save_param_step == 0:
				self.save_checkpoint(epoch, avg_epoch_vae_loss, avg_epoch_discriminator_loss, is_best=False)

	def validate(self):
		pass

	def save_results(self, epoch, img_recon):
		save_image(img_recon, os.path.join(self.log_result_dir, 'epoch_{}.png'.format(epoch)), nrow=8, normalize=True)
