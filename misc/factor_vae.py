import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms

from PIL import Image
import random
import numpy as np 

def kl_divergence(mu, logvar):
	kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
	return kld

def permute_dims(z):
	B,_ = z.size()
	perm_z = []

	for z_j in z.split(1,1):
		perm = torch.randperm(B).to(z.device)
		perm_z_j = z_j[perm]
		perm_z.append(perm_z_j)

	return torch.cat(perm_z,1)

class MNISTData(MNIST):
	def __init__(self, root, train=True, download=True, transform=None):
		super(MNISTData, self).__init__(root, transform=transform, train=train, download=download)
		self.indices = range(len(self))

	def __getitem__(self, index_1):
		index_2 = random.choice(self.indices)

		img_1, img_2 = self.data[index_1], self.data[index_2]

		img_1 = Image.fromarray(img_1.numpy(), mode='L')
		img_2 = Image.fromarray(img_2.numpy(), mode='L')

		if self.transform is not None:
			img_1 = self.transform(img_1)
			img_2 = self.transform(img_2)

		return img_1, img_2

class CustomImageFolder(ImageFolder):
	def __init__(self, root, transform=None):
		super(CustomImageFolder, self).__init__(root,transform)
		self.indices = range(len(self))

	def __getitem__(self, index_1):
		index_2 = random.choice(self.indices)

		path_1 = self.imgs[index1][0]
		path_2 = self.imgs[index2][0]
		img_1 = self.loader(path_1)
		img_2 = self.loader(path_2)
		if self.transform is not None:
			img_1 = self.transform(img_1)
			img_2 = self.transform(img_2)

		return img_1, img_2

class CustomTensorDataset(Dataset):
	def __init__(self, data_tensor, transform=None):
		# super()
		self.data_tensor = data_tensor
		self.transform = transform

		self.indices = range(self.data_tensor.size()[0])

	def __getitem__(self, index1):
		index2 = random.choice(self.indices)

		img1 = self.data_tensor[index1]
		img2 = self.data_tensor[index2]
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)

		return img1, img2

	def __len__(self):
		return self.data_tensor.size()[0]

def get_data(data_config):

	transform = transforms.Compose([
		transforms.Resize(data_config['img_size'],data_config['img_size']),
		transforms.ToTensor(),
		])

	if data_config['name'] == 'mnist':
		train_data = MNISTData(root=data_config['root'], train=True, download=True, transform=transform)
		val_data = MNISTData(root=data_config['root'], train=False, download=True, transform=transform)
	elif data_config['name'] == 'dsprites':
		data = np.load(data_config['root'], encoding='latin1')
		data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
		dataset = CustomTensorDataset(data, classification_label)
		train_data = dataset
		val_data = None

	else:
		train_data = CustomImageFolder(data_config['train_dir'], transform=transform)
		val_data = CustomImageFolder(data_config['val_dir'], transform=transform)

	return train_data, val_data
