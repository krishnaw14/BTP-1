import nnabla as nn 
import nnabla.functions as F 
import nnabla.solvers as S 

import numpy as np 
import os
from tqdm import trange

class GlowTrainer(object):

	def __init__(self, model, solver, train_loader, val_loader, config, 
		monitor_train_loss, monitor_train_recon, monitor_val_loss=None, monitor_val_recon=None, comm=None):
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.model = model 
		self.solver = solver 
		self.save_model_path = config['model']['save_path']
		self.warmup_iterations = config['train']['warmup_iterations']
		self.weight_decay = config['train']['weight_decay']

		self.train_iters_per_epoch = np.ceil(self.train_loader.size/self.train_loader.batch_size)
		self.val_iters_per_epoch = np.ceil(self.val_loader.size/self.val_loader.batch_size)

		self.monitor_train_loss = monitor_train_loss
		self.monitor_train_recon = monitor_train_recon
		self.monitor_val_loss = monitor_val_loss
		self.monitor_val_recon = monnitor_val_recon

		self.comm = comm

	def save_checkpoint(self, path, epoch):
		folder_path = os.path.join(self.save_model_path, "epoch_{}".format(epoch))
		os.makedirs(folder_path, exist_ok=True)
		nn.save_parameters(os.path.join(folder_path, "params.h5"))
		self.solver.save_states(os.path.join(folder_path, "solvers.h5"))

	def load_checkpoint(self, epoch):
		folder_path = os.path.join(self.save_model_path, "epoch_{}".format(epoch))
		nn.load_parameters(os.path.join(folder_path, "params.h5"))
		self.solver.load_states(os.path.join(folder_path, "solvers.h5"))

	def conver_to_var(self, x):

	def scale_back_var(self, x):

	def forward_pass(self, x):

	def reverse_pass(self):

	def save_samples(self):

	def train(self, epoch):
		epoch_loss = 0
		pbar = trange(self.train_iters_per_epoch, desc="Training at epoch_{epoch}", disable=self.comm.rank>0)
		for i in pbar:
			data = self.data_loader.next()
			log_p_sum, logdet, z = self.forward_pass(data)

			# Set batch loss value
			epoch_loss += loss.d

			self.solver.zero_grad()
			loss.backward()
			self.solver.set_parameters(nn.get_parameters(), reuse_state = True, retain=False)

			self.solver.weight_decay(self.weight_decay)
			self.solver.update()

		epoch_loss = epoch_loss/self.train_iters_per_epoch
		self.monitor_train_loss.add(epoch, loss)
		self.monitor_train_recon.add(epoch, img_recon.d)

	def validate(self):

	def warmup(self, iteration):

		if self.warmup_iterations == 0:
			return
		learning_rate = self.solver.learning_rate()
		learning_rate *= (iteration/self.warmup_iterations)
		self.solver.set_learning_rate(learning_rate)

