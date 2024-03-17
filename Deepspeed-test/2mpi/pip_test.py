from turtle import forward
import  deepspeed

from torch.utils.data import Dataset
import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch import optim

import  torchvision
from    matplotlib import pyplot as plt

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from    utils import *
import	const_for_mpi as c

c.update_constants()

if torch.cuda.is_available():
	gpu_num = torch.cuda.device_count()
	torch.cuda.set_device(0)
	device = torch.device('cuda')
	print(f"CUDA is available, running on {gpu_num} GPU(s)")
else:
	gpu_num = 0
	print("CUDA is not available. Running on CPU instead.")
	device = torch.device('cpu')

train = torchvision.datasets.MNIST(c.root_dir + 'mnist_data', train=True, download=True,
							transform=torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								torchvision.transforms.Normalize(
									(0.1307,), (0.3081,))
							]))

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x_processed = x.view(x.size(0), 28*28)
        y_processed = one_hot_num(y)  # Replace with your one_hot implementation
        return x_processed, y_processed

# Assuming train_dataset is the original dataset
custom_dataset = CustomDataset(train)

test_batch_size = 64
test_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST(c.root_dir + 'mnist_data', train=False, download=True,
							   transform=torchvision.transforms.Compose([
								   torchvision.transforms.ToTensor(),
								   torchvision.transforms.Normalize(
									   (0.1307,), (0.3081,))
							   ])),
	batch_size=test_batch_size, shuffle=False)

deepspeed.init_distributed()

from deepspeed.pipe import PipelineModule, LayerSpec
class ConnectedPipe(PipelineModule):
	def __init__(self, num_classes=10, **kwargs):
		self.num_classes = num_classes
		specs = [
			LayerSpec(nn.Linear, 28*28, 256),
			LayerSpec(nn.ReLU, inplace=True),
			LayerSpec(nn.Linear, 256, 64),
			LayerSpec(nn.ReLU, inplace=True),
			LayerSpec(nn.Linear, 64, num_classes),
		]
		super().__init__(layers=specs, loss_fn=F.mse_loss, **kwargs)

model = ConnectedPipe(num_stages = 2, partition_method="parameters")

# 使用 DeepSpeed 初始化模型
# 此处的 deepspeed_config 在 constants 中初始化
model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
	model				= model,
	model_parameters	= model.parameters(),
	config				= c.config_file_path,
	training_data		= custom_dataset,
)

'''
deepspeed 不负责搬运数据！
for batch in train_iter:
	# Assuming your batch contains input data and labels
	data, labels = batch
	print(data.device)  # This will print the device of the data tensor
	break  # Only check the first batch for demonstration purposes
print(next(model_engine.parameters()).device)
'''

train_loss = []
train_iter = iter(train_loader)
model_engine.train()
for epoch in range(c.EPOCH):
	loss = model_engine.train_batch(data_iter=train_iter)
	train_loss.append(loss.item())


if c.global_rank == 0:
	print("here it is!")
	plot_curve(train_loss)	



