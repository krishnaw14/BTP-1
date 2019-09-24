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

class ClassifierVAETrainer(Trainer):

	def __init__(self, vae_model, discriminator, config, device):
		super(ClassifierVAETrainer, self).__init__()

		self.gamma = config['model']['gamma']
		self.device = device
		self.classifier_loss = nn.BCELoss().cuda()

	def forward_pass(self, img, label):
		img_recon, mu, logvar, z, Cz = self.model(img)
		recon_loss = torch.mean((img_recon-img)**2)
		kld = kl_divergence(mu, logvar)

		classifier_loss = self.classifier_loss(Cz, label)

		total_loss = recon_loss + kld + self.gamma*classifier_loss

		return total_loss, recon_loss, classifier_loss

	def train(self):
		for epoch in range(self.num_epochs):
			pbar = tqdm(enumerate(self.train_loader), desc = 'training batch_loss')
			epoch_loss = 0.0
			for (nbatch, data) in pbar:
				img, label = data
				img, label = img.to(self.device), label.to(self.device)

				total_loss, recon_loss, classifier_loss = self.forward_pass(img, label)
				# if epoch%self.log_result_step == 0 and nbatch == 0:
				# 	self.save_results(epoch, img_recon)
				self.optimizer.zero_grad()
				total_loss.backward()
				self.optimizer.step()

				epoch_loss += total_loss

				pbar.set_description('Epoch: {}, Total Loss: {}, Classifier Loss {}, Recon Error: {}'.format(epoch, 
					total_loss.item(), classifier_loss.item(), recon_loss.item()))
				
			avg_epoch_loss = epoch_loss/(self.train_loader.dataset.data.shape[0]//self.train_loader.batch_size)

			print('Average Loss:', avg_epoch_loss)

			if epoch%self.val_step == 0:
				self.validate()
			if epoch%self.save_param_step == 0:
				self.save_checkpoint(epoch, avg_epoch_vae_loss, avg_epoch_discriminator_loss, is_best=False)

	def validate(self):
		pass

	def save_results(self, epoch, img_recon):
		save_image(img_recon, os.path.join(self.log_result_dir, 'epoch_{}.png'.format(epoch)), nrow=8, normalize=True)
