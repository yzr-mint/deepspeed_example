# 引入
- （仅供参考）
- 用MNIST手写数字任务来应用deepspeed
# 安装
```shell
pip install deepspeed
ds_report
```
- ![[Pasted image 20240124170010.png]]
- 这种情况下，需要在脚本文件中指定（一个高于5.0.0版本的）g++编译器的路径，以在运行时编译可能需要的选项（如融合的Adam优化器，如果优化器选用了Adam）
- 一些无法运行时编译的选项，WARNING中给出了原因。
- 由于手写数字识别任务并不需要这些无法运行时编译的选项，可以直接pip install
- 这样会安装一个0.13.0的deepspeed
- 如果加入这个可能可以预先打开所有选项
```shell
DS_BUILD_OPS=1
```
# 使用deepspeed（数据并行）
## 写deepspeed的设置文件（deeps.json）
```json
{
  "train_micro_batch_size_per_gpu": 64,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "SGD",
    "params": {
      "lr": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": 9000,
      "warmup_min_lr": 0.0001,
      "warmup_max_lr": 0.1,
      "warmup_num_steps": 1000,
      "warmup_type": "linear",
      "last_batch_iteration": -1
    }
  },
  "steps_per_print": 100,
  "fp16": {
    "enabled": false
  }
}
```
- scheduler指的是学习率更新机制。尝试过使用自定义的更新机制，暂未成功
- optimizer是优化器。
- 会把gradient_accumulation_steps个batch的结果一次更新
- "fp16"，"bp16"是混合精度开关，只能开启一个（且GPU支持这个浮点格式）
- https://www.deepspeed.ai/docs/config-json/
- https://www.deepspeed.ai/tutorials/zero/

## 修改py代码
```python
import argparse
config_file_path = 0
global_rank = 0
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
```
- 不接收`--local_rank`会导致报错。但local_rank指的是节点内线程的编号，也就是说，在我们的2节点，每个节点1个GPU的情况下，两个线程都接收到0，所以不用
- `global_rank = int(os.environ.get('RANK'))`会返回一个全局的线程编号。一定要转换成int
- `--deepspeed_config`是自己在调用的时候传入的参数，为了把刚才编辑的deepspeed的设置文件传进来

```python
model_engine, optimizer, train_iter, lr_scheduler = deepspeed.initialize(
	model				= model,
	model_parameters	= model.parameters(),
	config				= c.config_file_path,
	training_data		= train,
)
```
- 这样就初始化完成了（在低版本deepspeed里面会不一样，直接用新版本的就好）
```python
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
```
- deepspeed只负责搬运模型，不负责搬运数据。
- 不用手动置零梯度
- 此处因为两个节点上的唯一GPU都是编号0，所以没问题（我之前统一`torch.cuda.set_device(0)`）。不知道其他情况如何。

## pbs文件编写
```shell
#!/bin/bash -x
#PBS -N test_deepspeed
#PBS -l nodes=2:ppn=1:gpus=1
#PBS -j oe
#PBS -q gpu

> "/public/home/jiangy6/yezhr7/Deepspeed-test/hosts.txt"
while IFS= read -r line
do
  echo "$line slots=1" >> "/public/home/jiangy6/yezhr7/Deepspeed-test/hosts.txt"
done < $PBS_NODEFILE

source /public/software/profile.d/cuda11.6_cudnn8.9.sh
source /public/software/profile.d/compiler_gnu-7.2.0.sh
conda init
source ~/.bashrc
conda activate /public/home/jiangy6/yezhr7/envd
deepspeed --hostfile "/public/home/jiangy6/yezhr7/Deepspeed-test/hosts.txt" "/public/home/jiangy6/yezhr7/Deepspeed-test/Deepspeed_test.py" --deepspeed_config "/public/home/jiangy6/yezhr7/Deepspeed-test/deeps.json"
```
- 由于`$PBS_NODEFILE`返回的列表只有节点编号，没有slot(卡的数量)信息，所以处理了一下
- 把hostfile给deepspeed，把config文件给py就可以了
# pipline尝试
## 数据处理
```python
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
```
- 这里需要把数据集整理完成，没法（像先前一样）在训练时处理
- 不用指定搬运到设备上
- 注意匹配loss计算时的size，因为没法在计算loss之前处理
## 模型
```python
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
```
- `LayerSpec`的语法是：模型类，类初始化的参数
- 其余跟nn.Sequence类似
## 训练
```python
train_loss = []
train_iter = iter(train_loader)
model_engine.train()
for epoch in range(c.EPOCH):
	loss = model_engine.train_batch(data_iter=train_iter)
	train_loss.append(loss.item())
```
- 一次直接把整个数据集放进去
## json
```json
"pipeline": {
  "type": "simple",
  "params": {
    "num_stages": 2
  }
}
```
- 加一个条目即可，其余不用改动

# 用MPI来通信
## 安装mpi4py
```shell
pip install mpi4py
```
## 编写pbs脚本（要加入MPI环境）
```shell
#!/bin/bash -x
#PBS -N test_deepspeed
#PBS -l nodes=2:ppn=1:gpus=1
#PBS -j oe
#PBS -q gpu

ROOT="/public/home/jiangy6/yezhr7/Deepspeed-test/"
ENVIRONMENT="/public/home/jiangy6/yezhr7/envd"

> ${ROOT}"hosts.txt"
while IFS= read -r line
do
  echo "$line slots=1" >> ${ROOT}"hosts.txt"
done < $PBS_NODEFILE

source /public/software/profile.d/cuda11.6_cudnn8.9.sh
source /public/software/profile.d/compiler_gnu-7.2.0.sh
source /public/software/profile.d/mpi_openmpi-intel-2.1.2.sh
conda init
source ~/.bashrc
conda activate $ENVIRONMENT

mpirun -np 2 -hostfile ${ROOT}"hosts.txt" \
	-x UCX_TLS=openib \
	python ${ROOT}"pip_test.py" \
	--deepspeed_config ${ROOT}"deeps.json"

```
- py文件里面只用更改获取RANK的方式
```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
```
# -----------------
