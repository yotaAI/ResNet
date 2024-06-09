import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device ",device)

arc_38 = [[64,3],[128,4],[256,6],[512,3]]

class IdentityBlock(nn.Module):
    def __init__(self,f,in_filter,filters):
        super().__init__()
        Fin = in_filter
        F1,F2,F3 = filters
        self.conv1 = nn.Conv2d(Fin,F1,kernel_size=1,stride=(1,1),padding='valid')
        self.bn1   = nn.BatchNorm2d(F1, eps=1e-05)

        self.conv2 = nn.Conv2d(F1,F2,kernel_size=f,stride=(1,1),padding='same')
        self.bn2   = nn.BatchNorm2d(F2, eps=1e-05)

        self.conv3 = nn.Conv2d(F2, F3,kernel_size=1,stride=(1,1),padding='valid')
        self.bn3   = nn.BatchNorm2d(F3, eps=1e-05)
        
        self.relu  = nn.ReLU()
    def forward(self,X):
        X_shortcut = X

        X = self.relu(self.bn1(self.conv1(X)))
        X = self.relu(self.bn2(self.conv2(X)))
        X = self.bn3(self.conv3(X))

        X = X + X_shortcut
        X = self.relu(X)

        return X
        

class ConvolutionalBlock(nn.Module):
    def __init__(self,f,in_filter,out_filters,s=2):
        super().__init__()
        Fin = in_filter
        F1,F2,F3 = out_filters
        self.conv1 = nn.Conv2d(Fin,F1,kernel_size=1,stride=(s,s),padding='valid')
        self.bn1   = nn.BatchNorm2d(F1)

        self.conv2 = nn.Conv2d(F1,F2,kernel_size=1,stride=(1,1),padding='same')
        self.bn2   = nn.BatchNorm2d(F2)

        self.conv3 = nn.Conv2d(F2,F3,kernel_size=1,stride=(1,1),padding='valid')
        self.bn3   = nn.BatchNorm2d(F3)

        self.conv_sh = nn.Conv2d(Fin,F3,kernel_size=1,stride=(s,s),padding='valid')
        self.bn_sh   = nn.BatchNorm2d(F3)
        self.relu = nn.ReLU()

    def forward(self,X):
        X_shortcut = X
        X = self.relu(self.bn1(self.conv1(X)))
        X = self.relu(self.bn2(self.conv2(X)))
        X = self.bn3(self.conv3(X))

        X_shortcut = self.bn_sh(self.conv_sh(X_shortcut))

        X = self.relu(X + X_shortcut)
        return X


class ResNet(nn.Module):
    def __init__(self,input_channel=3,classes=1000,reps=[3,4,6,3],training=True):
        super().__init__()
        self.channe=input_channel
        self.classe = classes

        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu= nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.intermediate = nn.ModuleList()
        
        self.sequential = nn.Sequential()

        self.sequential.append(ConvolutionalBlock(3,64,out_filters = [64,64,256],s=1))
        for i in range(1,reps[0]):
            self.sequential.append(IdentityBlock(3,256,[64,64,256]))

        self.sequential.append(ConvolutionalBlock(3,256,out_filters = [128, 128, 512],s=2))
        for i in range(1,reps[1]):
            self.sequential.append(IdentityBlock(3,512,[128, 128, 512]))

        self.sequential.append(ConvolutionalBlock(3,512,out_filters = [256, 256, 1024],s=2))
        for i in range(1,reps[2]):
            self.sequential.append(IdentityBlock(3,1024,[256, 256, 1024]))

        self.sequential.append(ConvolutionalBlock(3,1024,out_filters = [512, 512, 2048],s=2))
        for i in range(1,reps[3]):
            self.sequential.append(IdentityBlock(3,2048,[512, 512, 2048]))
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7*7*2048,classes)
    def forward(self,X):
        X =  self.pool(self.relu(self.bn(self.conv1(X))))
        X = self.sequential(X)
        X = self.flatten(X)
        X = self.fc(X)
        return X



res_50_repeats = [3,4,6,3]

if __name__=='__main__':
	test = ResNet().to(device)
	print(torchsummary.summary(test,(3,224,224)))
