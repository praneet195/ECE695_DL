import torch
import torchvision
import torchvision.transforms as tf
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import pickle
import torch.nn as nn
#import torch.nn.functional as F
#import matplotlib.pyplot as plt
#import time
#import torch.optim as optim

#fig,ax=plt.subplots(1,1)
#ax.set_xlabel('Epoch')
#ax.set_ylabel('Loss')
#plt.ion()
#plt.show()
#losses=[]
#eps=[]

'''Dynamic Plot'''
#def plt_dynamic(x, y, ax, colors=['b']):
 #   for color in colors:
  #      ax.plot(x, y, color)
   #     plt.draw()
    #    plt.pause(0.001)
    #fig.canvas.draw()

'''Custom Cifar-10 with only Cats and Dogs, Code Used from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html'''
class CifarCatDog(Dataset):
    def __init__(self,root,labels,transforms=None,train=True,download=True): 
        self.train=train
        self.labels=labels
        self.transforms=transforms
        self.data=[]
        self.train_list=[]
        self.test_list=[]
        self.targets=[]
        self.root=root
        self.path=self.root+"/cifar-10-batches-py/"
        
        for i in range(1,6):
            self.train_list.append(self.path+"data_batch_"+str(i))
            
        self.test_list.append(self.path+"test_batch")

        if self.train:
            self.loop_list=self.train_list
        else:
            self.loop_list=self.test_list

        for f_name in self.loop_list:
            with open(f_name,"rb") as f:
                entity=pickle.load(f,encoding='latin1')
                for data,label in zip(entity['data'],entity['labels']):
                        if label in labels:
                            self.data.append(data)
                            if label==3:
                                self.targets.append([1,0])
                            else:
                                self.targets.append([0,1])

        self.data = np.vstack(self.data).reshape(-1,3,32,32)
        self.data=self.data.transpose((0,2,3,1))
        self.targets=np.vstack(self.targets).reshape(-1,2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img,target=self.data[idx],self.targets[idx]
        img=Image.fromarray(img)
        if self.transforms is not None:
            img =self.transforms(img)
        return img,target

    

#not using nn.Module, creating custom net
class _2_layer_net():
    def __init__(self):
        self.dtype=torch.float
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.D_in, self.H1, self.H2,self.D_out = (3*32*32), 1000,256, 2
        self.w1=torch.randn(self.D_in,self.H1,device=self.device,dtype=self.dtype)
        self.w2=torch.randn(self.H1,self.H2,device=self.device,dtype=self.dtype)
        self.w3=torch.randn(self.H2,self.D_out,device=self.device,dtype=self.dtype)
        self.w1=self.w1.to(self.device)
        self.w2=self.w2.to(self.device)
        self.w3=self.w3.to(self.device)
    
    def forward(self,inp):
        self.inp=inp.to(self.device)
        self.x=self.inp.view(self.inp.size(0),-1)
        self.x=self.x.float()
        self.z1=self.x.mm(self.w1)
        self.h1=self.z1.clamp(min=0)
        self.z2=self.h1.mm(self.w2)
        self.h2=self.z2.clamp(min=0)
        self.y_pred=self.h2.mm(self.w3)

    def loss_function(self,labels):
        self.labels=labels.to(self.device)
        self.loss=(1/len(self.labels))*(self.y_pred-self.labels).pow(2).sum().item()
        return self.loss

    def backprop(self):
            self.dy_pred=2*(self.y_pred-self.labels)
            self.dw3=self.h2.t().mm(self.dy_pred)
            self.dh2=self.dy_pred.mm(self.w3.t())
            self.dz2=self.dh2.clone()
            self.dz2[self.z2<0]=0
            self.dw2=self.h1.t().mm(self.dz2)
            self.dh1=self.dz2.mm(self.w2.t())
            self.dz1=self.dh1.clone()
            self.dz1[self.z1<0]=0
            self.dw1=self.x.t().mm(self.dz1)

    def update_weights(self,lr):
        self.w1-=lr*self.dw1
        self.w2-=lr*self.dw2
        self.w3-=lr*self.dw3

    def acc_function(self):
        acc=0
        for i in range(len(self.y_pred)):
            if torch.argmax(self.y_pred[i]) == torch.argmax(self.labels[i]):
                    acc+=1
        return acc

if __name__=="__main__":
    batch_size=4
    transforms = tf.Compose([tf.ToTensor(),tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    original_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True)
    
    cd_cifar10_train=CifarCatDog("./data",[3,5],transforms=transforms,train=True,download=True)
    cd_cifar10_test=CifarCatDog("./data",[3,5],transforms=transforms,train=False,download=True)

    trainloader=torch.utils.data.DataLoader(cd_cifar10_train,batch_size=batch_size,shuffle=True,num_workers=2)
    testloader=torch.utils.data.DataLoader(cd_cifar10_test,batch_size=batch_size,shuffle=True,num_workers=2)
    
    lr=1e-8
    epochs=100
    net=_2_layer_net()
    with open("output.txt","w") as f:
        for e in range(epochs):
            total_loss=0     
            total_acc=0
            iter_num=0
            for i, data in enumerate(trainloader):
                iter_num+=1
                inputs, labels=data
                net.forward(inputs)
                total_loss+=net.loss_function(labels)
                total_acc+=net.acc_function()
                net.backprop()
                net.update_weights(lr)
            f.write("Epoch "+str(e)+": "+ str(total_loss/float(iter_num))+"\n")
        f.write("\n")
        total_loss=0
        total_acc=0
        for i, data in enumerate(testloader):
            inputs,labels=data
            net.forward(inputs)
            total_loss+=net.loss_function(labels)
            total_acc+=net.acc_function()
        f.write("Test Accuracy: "+str((total_acc/len(cd_cifar10_test.data))*100.0)+"%")

