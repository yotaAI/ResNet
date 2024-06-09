import os,sys
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from resnet import res_50_repeats, ResNet
from dataset import ImageNetDataset
from utils import accuracy_calculate 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':

	EPOCH=100
	BATCH_SIZE=256
	LR=1e-2
	MOMENTUM=0.9
	L2_REG = 5e-3
	dataset_path = "./imagenet"
	INPUT_SHAPE=(224,224)
	loss_path = 'loss_resnet.txt'
	model_path = './resnet/'
	pretrained_model=None
	num_classes= 1000

	os.makedirs(model_path,exist_ok=True)

	#====================Initializing=================
	curr_epoch=-1
	calc=5

	training_dataset = ImageNetDataset(dataset_path,dataset_type='train')
	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

	test_dataset = ImageNetDataset(dataset_path,dataset_type='test')
	test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

	resnet = ResNet(3,num_classes,res_50_repeats).to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(resnet.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=L2_REG)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',0.1,2)

	if pretrained_model!=None:
		checkpoint=torch.load(pretrained_model)
		resnet.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		curr_epoch = checkpoint['epoch']
		print(f'Pretrained Epoch : {curr_epoch}')
		optimizer.param_groups[0]['lr'] = LR

	total_loss = []
	total_accuracy = []

	for epoch in range(curr_epoch+1,EPOCH):
		current_loss = []
		current_accuracy = []

		with tqdm(training_loader,ncols=150) as tepoch:
			for i,data in enumerate(tepoch):
				resnet.train()
				inputs,labels = data
				inputs = inputs.to(device)
				labels = labels.to(device)

				outputs = resnet(inputs)

				loss = loss_fn(outputs,labels)
				current_loss.append(loss.item())
				optimizer.zero_Grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					accuracy = accuracy_calculate(labels,outputs)/BATCH_SIZE
					current_accuracy.append(accuracy)

					if i%calc==0 and i>=calc:
						tepoch.set_description(f'EP {epoch}')
						tepoch.set_postfix(Loss = loss.item(), A = f'{accuracy:.4f}', LR = optimizer.param_groups[0]['lr'])

						if loss.item()==float('nan'):
							exit(0)

			with torch.no_grad():
				total_loss.append(np.average(current_loss))
				total_accuracy.append(np.average(torch.tensor(current_accuracy).cpu()))

				#======================== Testing==================

				test_accuracy = []
				test_loss = []
				resnet.eval()

				for (X_test,y_test) in tqdm(test_loader):
					pred = resnet(X_test.to(device))
					pred = pred.cpu()
					test_loss.append(loss_fn(pred,y_test))
					test_accuracy.append(accuracy_calculate(y_test,pred))

				total_test_loss = np.average(test_loss)
				total_test_accuracy = np.average(test_accuracy)

				print(f'Test Loss : {total_test_loss} Accuracy : {total_test_accuracy}')

				#===================== Scheduler ===================
				scheduler.step(total_test_loss)
				LR = optimizer.param_groups[0]['lr']

			with open(loss_path,'a+') as l:
				l.write(f'Epoch : {epoch} LR : {LR} LOSS : {np.average(current_loss)} Accuracy : {np.average(torch.tensor(current_accuracy).cpu())} Test Loss : {total_test_loss} Test Accuracy : {total_test_accuracy}\n')
		state_dict {
		'model_name' : "Resnet",
		'epoch' : epoch,
		'model' : resnet.state_dict(),
		'optimizer' : optimizer.state_dict(),
		}

		torch.save(state_dict,os.path.join(model_path,f'final_model.pt'))
		print('Model Saved . . .')
