import argparse
import os 
import torch

from trainers.vq_vae_pytorch import VQVAETrainer
from models.vq_vae_pytorch import VQVAE
from utils import read_yaml

def get_parser():
	parser = argparse.ArgumentParser(description='Generative Model trainings')
	parser.add_argument('--config_path', '-c', type=str, required=True,
		help='Path to config file for training and model parameters')
	parser.add_argument('--model', '-m', type=str, default='vqvae',
		help='Model Name to Train')
	# Add more options for inference
	return parser

if __name__ == '__main__':
	parser = get_parser()
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	config = read_yaml(args.config_path)
	if args.model -- 'vqvae':
		model = VQVAE(config).to(device)
		trainer = VQVAETrainer(model, config)
	else:
		model = Glow(config).to(device)
		trainer = GlowTrainer(model, config)
	
	trainer.train()