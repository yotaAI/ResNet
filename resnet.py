import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device ",device)

arc_38 = [[64,3],[128,4],[256,6],[512,3]]

class BasicBlock(nn.Module):
	def __init__(self,input_channel,output_channel):
		super().__init__()
		self.conv_block = nn.Sequential()
		if input_channel==output_channel:
			self.conv_block.append(nn.Conv2d(input_channel,output_channel,kernel_size=3,stride=1,padding='same'))
		else:
			self.conv_block.append(nn.Conv2d(input_channel,output_channel,kernel_size=3,stride=2,padding=1))

		self.conv_block.append(nn.BatchNorm2d(output_channel))
		self.conv_block.append(nn.ReLU())

		self.conv_block.append(nn.Conv2d(output_channel,output_channel,kernel_size=3,stride=1,padding='same'))
		self.conv_block.append(nn.BatchNorm2d(output_channel))

		self.skip_block = None
		if input_channel!=output_channel:
			self.skip_block = nn.Conv2d(input_channel,output_channel,kernel_size=1,stride=2,padding='valid')

		self.relu = nn.ReLU()
		self.batch_norm = nn.BatchNorm2d(output_channel)

	def forward(self,x):
		skip = x
		x = self.conv_block(x)
		if self.skip_block !=None:
			skip = self.skip_block(skip)
		x = x + skip

		return self.relu(self.batch_norm(x))

class Resnet(nn.Module):
	def __init__(self,input_channel,architecture,num_classes=1000):
		super().__init__()

		self.input_channels = input_channel
		self.architecture = architecture

		self.inp_layer = nn.Sequential(
			nn.Conv2d(self.input_channels,64,stride=2,padding=3,kernel_size=7),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
			)

		self.resnet_blocks = nn.ModuleList()

		input_channel=64
		for (channel,times) in self.architecture:
			for _ in range(times):
				self.resnet_blocks.append(BasicBlock(input_channel,channel))
				input_channel=channel
		self.flatten = nn.Flatten()
		self.fc = nn.Linear(7*7*512,num_classes)

	def forward(self,x):
		x = self.inp_layer(x)
		for residual in self.resnet_blocks:
			x = residual(x)
		x = self.fc(self.flatten(x))
		return x

if __name__=='__main__':
	resnet = Resnet(3,arc_38).to(device)
	print(torchsummary.summary(resnet,(3,224,224)))
