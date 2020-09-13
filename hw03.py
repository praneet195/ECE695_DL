import torch.nn as nn
import torch
import torchvision
import numpy as np
import sys
import os
import torchvision.transforms as tf
import torch.nn.functional as F

class TemplateNet1(nn.Module):
    def __init__(self):
        super(TemplateNet1,self).__init__()
        self.conv1 = nn.Conv2d(3,128,3,padding=1)
        self.conv2 = nn.Conv2d(128,128,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(6272,1000)
        self.fc2 = nn.Linear(1000,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,6272)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TemplateNet2(nn.Module):
    def __init__(self):
        super(TemplateNet2,self).__init__()
        self.conv1 = nn.Conv2d(3,128,3,)
        self.conv2 = nn.Conv2d(128,128,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4608,1000)
        self.fc2 = nn.Linear(1000,10)

    def forward(self,x): 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,4608)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

class TemplateNet3(nn.Module):
    def __init__(self):
        super(TemplateNet3,self).__init__()
        self.conv1 = nn.Conv2d(3,128,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(28800,1000)
        self.fc2 = nn.Linear(1000,10)

    def forward(self,x): 
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1,28800)
        #6272,4608,28800
        #x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

def blockPrint():
    sys.stdout=open(os.devnull,'w')

def enablePrint():
    sys.stdout=sys.__stdout__

def run_code_for_training(net):
    batch_size=4
    epochs=1
    transforms = tf.Compose([tf.ToTensor(),tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    blockPrint()
    train_data=torchvision.datasets.CIFAR10(".", train=True, transform=transforms, target_transform=None, download=True)
    enablePrint()
    train_data_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=2)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=1e-3,momentum=0.9)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i%2000 == 1999:
                #print("[epoch:%d, batch:%5d] loss: %.3f" %(epoch + 1, i + 1, running_loss / float(2000)))
                if i==11999:
                    print("[epoch:%d, batch:%5d] loss: %.3f" %(epoch + 1, i + 1, running_loss / float(2000)))
                    with open("output.txt","a+") as f:
                        f.write("[epoch:%d, batch:%5d] loss: %.3f\n" %(epoch + 1, i + 1, running_loss / float(2000)))
                running_loss=0.0



def confusion_matrix(net):
    classes=10
    batch_size=4
    confusion_matrix = torch.zeros(classes,classes)
    transforms = tf.Compose([tf.ToTensor(),tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    blockPrint()
    test_data=torchvision.datasets.CIFAR10(".", train=False, transform=transforms, target_transform=None, download=True)
    enablePrint()
    test_data_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True,num_workers=2)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, data in enumerate(test_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, preds =  torch.max(outputs,1)
        for t, p in zip(labels.view(-1),preds.view(-1)):
            confusion_matrix[t.long(),p.long()]+=1
    print(confusion_matrix)
    with open("output.txt","a+") as f:
        f.write(str(confusion_matrix))



if __name__=="__main__":
     torch.manual_seed(0)
     np.random.seed(0)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark =  False
     net3=TemplateNet3()
     net2=TemplateNet2()
     net1=TemplateNet1()
     run_code_for_training(net3)
     run_code_for_training(net2)
     run_code_for_training(net1)
     confusion_matrix(net1)

