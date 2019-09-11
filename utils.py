import yaml
def read_yaml(path):
	with open(path, 'r') as stream:
		try:
			return yaml.load(stream)
		except yaml.YAMLError as exc:
			return exc