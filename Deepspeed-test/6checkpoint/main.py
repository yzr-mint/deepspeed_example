import  deepspeed
import  torch.distributed

import	torch
from	torch import nn
from	torch.nn import functional as F

import	torchvision
import	pprint

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from    utils import plot_image, plot_curve, one_hot
import	constants as c

c.update_constants()
checkpoint_interval = 10
save_root = "/public/home/jiangy6/yezhr7/Deepspeed-test/checkpoint/save/"

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

load = True
train_loss = []
start_epoch = 0
# 恢复训练
if load:
	loaded_state, client_state = model_engine.load_checkpoint(load_dir=save_root)
	if loaded_state is not None:
		if c.global_rank == 0:
			pprint.pprint(loaded_state)
			pprint.pprint(client_state)
		start_epoch = client_state.get('epoch', 0)
		train_loss = client_state.get('losses', [])
'''
if c.global_rank == 0:
	pprint.pprint(optimizer.state_dict())
'''

model_engine.train()
for epoch in range(start_epoch, c.EPOCH):
	for batch_idx, (x, y) in enumerate(train_iter):

		x = x.to(model_engine.local_rank).view(x.size(0), 28*28)
		y = one_hot(y).to(model_engine.local_rank).half()
		
		out = model_engine(x)

		loss = F.mse_loss(out, y)

		model_engine.backward(loss)
		model_engine.step()
		lr_scheduler.step()
		
		
		train_loss.append(loss.item())
	if (epoch + 1) % checkpoint_interval == 0:
		client_state = {
				  'losses': train_loss,
				  'epoch': epoch + 1,
				  }
		if c.global_rank == 0:
			pprint.pprint(optimizer.state_dict())
		model_engine.save_checkpoint(save_dir=save_root, tag=None, client_state=client_state)

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


