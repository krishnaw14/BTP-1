from .base import Trainer
from misc.classifier_vae import kl_divergence, permute_dims, get_data

from tqdm import tqdm
import os
import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision.utils import save_image

class ClassifierVAETrainer(Trainer):

	def __init__(self, model, classifier, config, device):
		super(ClassifierVAETrainer, self).__init__(model, config, device)

		if config['data']['name'] != 'mnist':
			self.train_loader = self.get_data_loader(config['data'])
			self.iters_per_epoch = np.ceil(self.train_loader.dataset.data.shape[0]/self.train_loader.batch_size)

		self.gamma_init = config['model']['gamma']
		self.gamma_update_rate = config['model']['gamma_update_rate']
		self.device = device
		self.classifier_loss = nn.BCELoss().cuda()
		self.classifier = classifier
		self.optim_classifier = self.get_optimizer(classifier.parameters(), config['train']['optim'], config['train']['c_lr'])

		self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,20], gamma=0.5)
		self.identity = torch.eye(10).cuda()

	def get_data_loader(self, data_config, train=True):
		if train:
			self.train_dataset, self.val_dataset = get_data(data_config)
			return DataLoader(self.train_dataset, batch_size=data_config['batch_size'],
				shuffle=True, pin_memory=True)
		else:
			return DataLoader(self.val_dataset, batch_size=data_config['batch_size'],
				shuffle=True, pin_memory=True)

	def update_gamma(self, recon_loss):
		self.gamma = self.gamma_init
		self.gamma = self.gamma_init - self.gamma_update_rate*recon_loss

	def latent_traversal(self):
		self.model.eval()
		sample_idx = 2048
		interpolation_vals = torch.arange(-3,3, 2/3)
		img, y = self.val_loader.dataset.__getitem__(sample_idx)
		img = img.unsqueeze(0)
		y = self.identity[y].unsqueeze(0)
		img = img.to(self.device)
		img_z = self.model.encoder(img)[:, :self.model.z_dim]

		samples = []
		for row in range(self.model.z_dim):
			z_ = img_z.clone()
			for val in interpolation_vals:
				z_[:, row] = val
				sample = self.model.decoder(z_ + self.model.label_layer(y).unsqueeze(-1).unsqueeze(-1))
				sample = torch.sigmoid(sample)
				samples.append(sample)
		samples = torch.cat(samples, dim=0).cpu()

		save_image(samples, os.path.join(self.log_result_dir, 'Traverse_{}.png'.format(sample_idx)), nrow=9, normalize=True)


	def forward_pass(self, img, label, classify=True):
		# y = self.identity[label]
		y = label
		img_recon, mu, logvar, z, Cz = self.model(img, y)
		# recon_loss = torch.mean((img_recon-img)**2)
		recon_loss = F.binary_cross_entropy_with_logits(img_recon, img, size_average=False).div(img.size(0))
		# recon_loss = self.recon_loss(img_recon, img)
		kld = kl_divergence(mu, logvar)

		vae_loss = recon_loss + kld

		if classify:
			# Cz = self.classifier(z)
			# import pdb; pdb.set_trace()
			classifier_loss = self.classifier_loss(Cz, label.float())
		else:
			classifier_loss = 0

		# total_loss = recon_loss + kld + self.gamma*classifier_loss

		# import pdb; pdb.set_trace()

		return recon_loss, vae_loss, img_recon, z, classifier_loss

	def train(self):
		for epoch in range(self.num_epochs):
			pbar = tqdm(enumerate(self.train_loader), desc = 'training batch_loss', total = self.iters_per_epoch)
			epoch_loss = 0.0
			for (nbatch, data) in pbar:
				img, label = data
				img, label = img.to(self.device), label.to(self.device)

				if epoch == 0:
					classify = True
				else:
					classify = True
				recon_loss, vae_loss, img_recon, z, classifier_loss = self.forward_pass(img, label, classify)
			# 	if epoch%self.log_result_step == 0 and nbatch == 0:
			# 		self.save_results(epoch, img_recon)

				self.update_gamma(recon_loss.item())
				# classifier_loss = self.gamma*classifier_loss
				total_loss = vae_loss + self.gamma*classifier_loss

				self.optimizer.zero_grad()
				total_loss.backward(retain_graph=False)
				self.optimizer.step()

			# 	# if classify:
			# 	# 	self.optim_classifier.zero_grad()
			# 	# 	classifier_loss.backward(retain_graph=False)
			# 	# 	self.optim_classifier.step()

				epoch_loss += total_loss

				pbar.set_description('Epoch: {}, Total Loss: {}, Classifier Loss {}, Recon Error: {}, gamma: {}'.format(epoch, 
					total_loss.item(), classifier_loss.item(), recon_loss.item(), self.gamma))
				# # import pdb; pdb.set_trace()
			# # self.lr_scheduler.step()
				
				
			avg_epoch_loss = epoch_loss/(self.iters_per_epoch)

			print('Average Loss:', avg_epoch_loss.item())
			print('Reconstruction Loss', recon_loss.item())
			print('Classifier Loss:', classifier_loss)

			if epoch%self.val_step == 0:
				self.validate()
			if epoch%self.save_param_step == 0:
				self.save_checkpoint(epoch, avg_epoch_loss, is_best=False)

		# import pdb; pdb.set_trace();
		self.latent_traversal()


	def validate(self):
		pass

	def save_results(self, epoch, img_recon):
		save_image(img_recon, os.path.join(self.log_result_dir, 'epoch_{}.png'.format(epoch)), nrow=8, normalize=True)

