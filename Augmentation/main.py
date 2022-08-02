import configparser
from augment import call_augmentation_functions

def main():
	config = configparser.ConfigParser()
	config.read('./config/config.ini')
	aug_types = {'blur': config['AUGMENTATION']['Blur'], \
				'bright': config['AUGMENTATION']['Bright'], \
				'warp' : config['AUGMENTATION']['Warp'], \
				'rotate': config['AUGMENTATION']['rotate'], \
				'flip': config['AUGMENTATION']['Flip']
				}
	data_path = config['DATASET_PATH']['images_path']
	call_augmentation_functions(aug_types, data_path, config)

if __name__ == '__main__':
	main()