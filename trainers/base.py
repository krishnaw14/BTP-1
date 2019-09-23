import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):

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

	def forward_pass(self):
		pass	

	def save_results(self):
		pass

	def train(self):
		pass

	def validate(self):
		pass