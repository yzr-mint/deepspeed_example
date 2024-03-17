import  deepspeed

import	torch
from	torch import nn
from	torch.nn import functional as F
from	torch import optim

import	torchvision
import	math

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
									(0.1307,), (0.3081,))
							]))

test_batch_size = 64
test_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST(c.root_dir + 'mnist_data', train=False, download=True,
							   transform=torchvision.transforms.Compose([
								   torchvision.transforms.ToTensor(),
								   torchvision.transforms.Normalize(
									   (0.1307,), (0.3081,))
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

lrs = []
class CosineAnnealingLR:
	def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
		self.optimizer = optimizer
		self.T_max = T_max
		self.eta_min = eta_min
		self.last_epoch = last_epoch

	def get_lr(self):
		base_lr = self.optimizer.param_groups[0]['lr']
		if isinstance(base_lr, (float, int)):
			lrs.append(base_lr)
			# If the learning rate is a single float or int, return it as a list
			return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2]
		else:
			lrs.append(base_lr[0])
			# If the learning rate is iterable, apply cosine annealing to each element
			return [self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
					for lr in base_lr]		


	def step(self):
		self.last_epoch += 1
		new_lr = self.get_lr()

		for param_group, lr in zip(self.optimizer.param_groups, new_lr):
			param_group['lr'] = lr

	# 这两个函数是为了在中断，继续训练的时候维护lr调度器的状态。
	# 比如，此处是为了维护 self.last_epoch。这是在训练的时候会变化的变量
	def state_dict(self):
		return {'last_epoch': self.last_epoch}

	def load_state_dict(self, state_dict):
		self.last_epoch = state_dict['last_epoch']

optimizer = optim.SGD(model.parameters(), lr=0.1)

model_engine, optimizer, train_iter, lr_scheduler = deepspeed.initialize(
	model				= model,
	model_parameters	= model.parameters(),
	config				= c.config_file_path,
	training_data		= train,
	optimizer			= optimizer,
	lr_scheduler		= CosineAnnealingLR(optimizer, T_max=50, eta_min=0.01),
)


train_loss = []
model_engine.train()
for epoch in range(c.EPOCH):
	for batch_idx, (x, y) in enumerate(train_iter):
		x = x.to("cuda")
		out = model_engine(x.view(x.size(0), 28*28))

		loss = F.mse_loss(out, one_hot(y).to("cuda"))
		
		model_engine.backward(loss)
		model_engine.step()
		lr_scheduler.step()
		
		train_loss.append(loss.item())


if c.global_rank == 0:
	print("here it is!")
	# 我们关心学习率的变化，所以打印学习率
	plot_curve(train_loss)	
	
	#准确度测试
	model_engine.eval()
	total_correct = 0
	for x,y in test_loader:
		x = x.to("cuda")
		x = x.view(x.size(0), 28*28)
		out = model_engine(x)
		pred = out.argmax(dim=1)
		correct = pred.eq(y.to("cuda")).sum().float().item()
		total_correct += correct

	total_num = len(test_loader.dataset)
	acc = total_correct / total_num
	print('test acc:', acc)
	
	#以下部分是在打印，打印一个图片与对应的预测结果
	x, y = next(iter(test_loader))
	xdev = x.to("cuda")
	out = model_engine(xdev.view(x.size(0), 28*28))
	pred = out.argmax(dim=1)

	plot_image(x, pred, 'test')

