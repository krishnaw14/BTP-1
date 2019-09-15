import torch
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

class ActNorm(nn.Module):
	def __init__(self, config):

	def forward():

class InvConv2d(nn.Module):
	def __init__(self, channel):
		super(InvConv2d, self).__init__()

		weight = torch.randn(channel, channel)
		weight, _ = torch.qr(weight)
		weight = weight.unsqueeze(2).unsqueeze(3)
		weight = nn.Parameter(weight)

	def forward(self, x):
		_, _, H, W = x.shape
		out = F.conv2d(x, self.weight)
		logdet = (H*W*torch.slogdet(self.weight.squeeze().double())[1].float())
		return out, logdet

	def reverse(self, x):
		out = F.conv2d(x, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
		return out

class ZeroConv2d(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(ZeroConv2d,self).__init__()

		self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3)
		self.conv.weight.data.zero_()
		self.conv.bias.data.zero_()
		self.scale = nn.Parameter(torch.zeros(1,out_channel,1,1))

	def forward(self, x):
		out = F.pad(x, [1,1,1,1], value=1)
		out = self.conv(out)
		out = out*torch.exp(self.scale*3)

		return out

class AffineCoupling(nn.Module):
	def __init__(self, in_channel, filter_size=512, affine=True):
		super(AffineCoupling, self).__init__()

		self.affine = affine
		self.net = nn.Sequential(
			nn.Conv2d(in_channel//2, filter_size, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(filter_size, filter_size, 1),
			nn.ReLU(inplace=True),
			ZeroConv2d(filter_size, in_channel if self.affine else in_channel//2)
			)
		self.net[0].weight.data.normal_(0,0.05)
		self.net[0].weigt.data.zero_()
		self.net[2].weight.data.normal_(0,0.05)
		self.net[2].weigt.data.zero_()

	def forward(self):

class FlowStep(nn.Module):
	def __init__(self, config):

	def forward():

class Glow(nn.Module):
	def __init__(self, config):

	def forward():