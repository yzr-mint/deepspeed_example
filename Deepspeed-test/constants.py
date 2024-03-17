import argparse

EPOCH = 10

config_file_path = 0
global_rank = 0
root_dir = "/public/home/jiangy6/yezhr7/Deepspeed-test/"

import os

def parse_args():
	inputp = argparse.ArgumentParser(description="Input")
	inputp.add_argument('--deepspeed_config', type=str, default=None, 
						help='Path to DeepSpeed config file')
	inputp.add_argument('--local_rank', type=int, default=None, 
						help='Local_rank for deepspeed, inside the node')
	args = inputp.parse_args()
	return args

def update_constants():
	args = parse_args()
	
	if args.deepspeed_config is not None:
		global config_file_path
		config_file_path = args.deepspeed_config
		global global_rank
		global_rank = int(os.environ.get('RANK'))


	