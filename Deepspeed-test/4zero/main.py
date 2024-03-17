import  deepspeed
import  torch.distributed

import	torch
from	torch import nn
from	torch.nn import functional as F

import	torchvision

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from    utils import plot_image, plot_curve, one_hot
import	constants as c

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
									(0.1307,), (0.3081,)),	
							]))

test_batch_size = 64
test_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST(c.root_dir + 'mnist_data', train=False, download=True,
								transform=torchvision.transforms.Compose([
									torchvision.transforms.ToTensor(),
									torchvision.transforms.Normalize(
										(0.1307,), (0.3081,)),  
							   ])),
	batch_size=test_batch_size, shuffle=False)

#这是建立神经网络
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28*28, 256)
		self.fc2 = nn.Linear(256, 64)
		self.fc3 = nn.Linear(64, 10)
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
model = Net()

# 使用 DeepSpeed 初始化模型
# 此处的 deepspeed_config 在 constants 中初始化
model_engine, optimizer, train_iter, lr_scheduler = deepspeed.initialize(
	model				= model,
	model_parameters	= model.parameters(),
	config				= c.config_file_path,
	training_data		= train
)

train_loss = []
model_engine.train()
for epoch in range(c.EPOCH):
	for batch_idx, (x, y) in enumerate(train_iter):

		x = x.to(model_engine.local_rank).view(x.size(0), 28*28)
		y = one_hot(y).to(model_engine.local_rank).half()
		
		out = model_engine(x)

		loss = F.mse_loss(out, y)

		model_engine.backward(loss)
		model_engine.step()
		lr_scheduler.step()
		
		train_loss.append(loss.item())

	if c.global_rank == 0:	
		current_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为GB
		max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转换为GB
		torch.cuda.reset_peak_memory_stats()  # 重置最大显存统计，为下一个epoch准备
		print(f'Epoch {epoch+1}/{c.EPOCH}:')
		print(f'当前显存使用:			{current_memory_allocated:.2f} GB')
		print(f'本epoch最大显存使用:	{max_memory_allocated:.2f} GB')


if c.global_rank == 0:
	print("here it is!")
	plot_curve(train_loss)	

model_engine.eval()
total_correct = 0
for x,y in test_loader:
	x = x.to(model_engine.local_rank).view(x.size(0), 28*28)
	with torch.no_grad():
		out = model_engine(x)
	pred = out.argmax(dim=1)
	correct = pred.eq(y.to(model_engine.local_rank)).sum().float().item()
	total_correct += correct

if c.global_rank == 0:
	total_num = len(test_loader.dataset)
	acc = total_correct / total_num
	print('test acc:', acc)

x, y = next(iter(test_loader))
xdev = x.to(model_engine.local_rank)
out = model_engine(xdev.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
if c.global_rank == 0:
	plot_image(x.cpu(), pred.cpu(), 'test')


