''' This code is taken from DLStudio.py

Version:  1.0.7
   
Original Author: Avinash Kak (kak@purdue.edu).
I do not own the code. I have only made slight modifications to the already existing BMENet and SkipBlock that are available in the DLStudio module to try different SkipBlocks and chaneg the depth of the network accordingly.

Modifications are as listed:
    [1] 5 types of Skip blocks can be used
    [2] Both Cifar10 and ImageNet can be used
    [3] BMENet deeper variants. (Bug Fix???)
    [4] All torch resnet models can be run
    [5] ImageNet downloader has been uploaded  by me to piazza. It has to be incorporated into this.
    [6] Early Stopping
    '''

import torch
import torchvision.transforms as tf
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch.nn as nn
import os
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchsummary import summary
import torchvision.models as models
import argparse
import time

parser =  argparse.ArgumentParser(description="Playing with Skip Connections using the Cifar-10 or ImageNet Dataset")
parser.add_argument('-run',type=int,default=0,help='Set to 1 if you want to run testing or 0 for training, default is 0')
parser.add_argument('-model',type=str,default=None,help='If testing, provide the model, default is None')
parser.add_argument('-name',type=str,default=None,help="If testing, please provide model name to append results file name with model name, default is None")
parser.add_argument('-cifar10',type=int,default=1,help='Set to 1 if you want to run on Cifar10 dataset, default is 1')
parser.add_argument('-imagenet',type=int,default=0,help='Set to 1 if tou want to run on 5 class Imagenet, default is 0')
parser.add_argument('-imagenet_train',type=str,default=None,help='Path to the Imagenet Train Set, please ensure images are separated class wise because class_id is assigned using folder name')
parser.add_argument('-imagenet_test',type=str,default=None,help='Path to the Imagenet Test Set, please ensure images are separated class wise because class_id is assigned using this')
parser.add_argument('-batch_size',type=int,default=4,help="Set batch size, default value is 4")
parser.add_argument('-epochs',type=int,default=10,help="Set number of epochs, default value is 10")
parser.add_argument('-lr',type=float,default=1e-5,help="Set learning rate, default value is 1e-3")
parser.add_argument('-mtm',type=float,default=0.9,help="Set momentum, default value is 0.9")
parser.add_argument('-resize',type=int,default=32,help="Resize image Height x Width where Height = Width, default is 32 x 32")
parser.add_argument('-skipblock',type=int,default=0,help="Type of Skipblock incorporated, 0 for basic DLStudio block or ReLU Before addition, 1 for Original Residual Block,2 for BN after addition, 3 for ReLU-only pre-activation, 4 for CNN in identity path, by default uses SkipBlock 0")
parser.add_argument('-torch_resnet', type=str, default=None, help ="Run on original Torch Resnet Models, options: resnet18, resnet34, resnet50, resnet101, resnet152, resnext50, resnext101, wide_resnet50, wide_resnet101, by default runs on resnet18")
parser.add_argument('-depth',type=int,default=4,help="Deeper BMENet Variant, this paramter specifies depth, Depth of 4 is original")
parser.add_argument('-early_stop',type=int,default=0,help="If you want early stopping to be enabled")
args = parser.parse_args()
#classes={"cat":0,"dog":1,"bicycle":2,"truck":3,"boat":4}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def early_stop(max_val_acc,cur_val_acc,min_val_loss,cur_val_loss):
    if (max_val_acc - cur_val_acc) >= 5 and (cur_val_loss - min_val_loss) >= 0.2 :
        return True
    else:
        return False

class ImageNetClass(Dataset):
    def __init__(self,root,transforms=None,resize=(32,32)):
        self.transforms=transforms
        self.data=[]
        self.targets=[]
        self.root=root
        self.path=self.root
        self.resize=resize

        for class_id,i in enumerate(os.listdir(self.root)):
            for j in os.listdir(self.root+i):
                self.targets.append(class_id)
                img_path=self.root+i+"/"+j
                im=Image.open(img_path)
                im=im.convert("RGB")
                im=im.resize(self.resize)
                im=np.asarray(im)
                self.data.append(im)

        self.data = np.vstack(self.data).reshape(-1,3,resize[0],resize[1])
        self.data=self.data.transpose((0,2,3,1))
        self.targets=np.vstack(self.targets)
        print("Dataset Image Shape :"+str(self.data.shape))
        print("Dataset Labels Shape :"+str(self.targets.shape))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img,target=self.data[idx],self.targets[idx]
        if self.transforms is not None:
            img =self.transforms(img)
        return img,target


class SkipConnections(nn.Module):
    class SkipBlock(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(SkipConnections.SkipBlock, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            self.convo_1 = nn.Conv2d(in_ch,out_ch//2,1)
            self.convo_2=nn.Conv2d(out_ch//2,out_ch//2,3,padding=1,stride=1)
            self.convo_3=nn.Conv2d(out_ch//2,in_ch,1)
            norm_layer = nn.BatchNorm2d
            self.bn = norm_layer(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
        def forward(self, x):
            if args.skipblock==1:
                #Original Block not DLStudio
                identity = x                            
                out = self.convo(x)                      
                out = self.bn(out)                       
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)             
                    out = self.bn(out)                   
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                  
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                out = torch.nn.functional.relu(out)
            elif args.skipblock==2:
                #Addition then BatchNorm and ReLU
                identity = x
                out = self.convo(x)
                out = self.bn(out)
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                out = self.bn(out)
                out = torch.nn.functional.relu(out)
            elif args.skipblock==3:
                #ReLU only pre-activation
                identity = x
                out = torch.nn.functional.relu(x)
                out = self.convo(out)
                out = self.bn(out)
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)
                    out = self.bn(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
            elif args.skipblock==4:
                #CNN in Identity Path
                identity = x
                out = self.convo(x)
                out = self.bn(out)
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)
                    out = self.bn(out)
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    identity = self.convo_3(self.convo_2(self.convo_1(identity)))
                    if self.in_ch == self.out_ch:
                        out += identity
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                out=self.bn(out)
                out = torch.nn.functional.relu(out)
                
            else:
                #DLStudio Original, ReLu Before Addtion
                identity = x
                out = self.convo(x)
                out = self.bn(out)
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)
                    out = self.bn(out)
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
            return out


    class BMEnet(nn.Module):
            def __init__(self, skip_connections=True, depth=args.depth):
                super(SkipConnections.BMEnet, self).__init__()
                self.pool_count = 3
                self.depth = depth
                self.bn = nn.BatchNorm2d(64)
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.skip64 = SkipConnections.SkipBlock(64, 64, skip_connections=skip_connections)
                self.skip64list = nn.ModuleList([SkipConnections.SkipBlock(64, 64, skip_connections=skip_connections) for i in range(self.depth)])
                self.skip64ds = SkipConnections.SkipBlock(64, 64,
                                                downsample=True, skip_connections=skip_connections)
                self.skip64to128 = SkipConnections.SkipBlock(64, 128,
                                                                skip_connections=skip_connections )
                self.skip128 = SkipConnections.SkipBlock(128, 128, skip_connections=skip_connections)
                self.skip128list = nn.ModuleList([SkipConnections.SkipBlock(128, 128, skip_connections=skip_connections)
 for i in range(self.depth)])
                self.skip128ds = SkipConnections.SkipBlock(128,128,
                                                downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(128 * (args.resize // 2**self.pool_count)**2, 2048)
                self.fc2 = nn.Linear(2048,512)
                if args.imagenet:
                    self.fc3 = nn.Linear(512,5)
                else:
                    self.fc3 = nn.Linear(512,10)

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.bn(self.conv(x))))
                for skip64 in self.skip64list:
                    x = skip64(x)
                x = self.skip64ds(x)
                for skip64 in self.skip64list:
                    x = skip64(x)
                x = self.skip64to128(x)
                for skip128 in self.skip128list:
                    x = skip128(x)
                x = self.skip128ds(x)
                for skip128 in self.skip128list:
                    x = skip128(x)
                x = x.view(-1, 128 * (args.resize // 2**self.pool_count)**2 )
                #x = torch.flatten(x,start_dim=1) )
                x = torch.nn.functional.relu(self.fc1(x))
                x = torch.nn.functional.relu(self.fc2(x))
                x = self.fc3(x)
                return x

def run_code_for_training(net,model_name):
    transforms = tf.Compose([tf.ToTensor(),tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    if args.imagenet:
        args.resize=256
        assert args.imagenet_train, "ImageNet Train Dataset path not specified"
        print("Loading ImageNet Train Set............")
        dataset = ImageNetClass(args.imagenet_train,transforms=transforms,resize=(args.resize,args.resize))
    else:    
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms)

                                        #download=True, transform=transforms)
    train_len=int(0.85*len(dataset))
    val_len=len(dataset)-train_len
    train_set, val_set= random_split(dataset,[train_len,val_len])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                          shuffle=True, num_workers=2)
                                       
    valloader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                          shuffle=True, num_workers=2)
                                       
    num_batches_train=int(len(train_set)/args.batch_size)
    num_batches_val=int(len(val_set)/args.batch_size)
    net=net.to(device)
    summary(net,input_size=(3,args.resize,args.resize))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    if args.imagenet:
            result_file=model_name+"_imagenet.txt"
    else:
            result_file=model_name+"_cifar10.txt"
    max_val_acc=0
    min_val_loss=9999
    with open(result_file,"w") as f:
        for e in range(args.epochs):
            train_loss=0
            val_loss = 0
            train_correct=0
            val_correct=0
            batch_num=0
            for i,data in enumerate(trainloader):
                batch_num+=1
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                labels=torch.squeeze(labels)
                if labels.dim()==0:
                    loss=criterion(outputs,labels.unsqueeze(0))
                else:
                    loss = criterion(outputs,labels)
                labels=labels.detach().cpu().numpy()
                outputs=outputs.detach().cpu().numpy()
                try:
                    res= np.argmax(outputs,axis=1)
                    res=res.reshape(-1,1)
                    labels=labels.reshape(-1,1)
                    res=labels==res
                    for i in res:
                        if i:
                            train_correct+=1
                except:
                    if np.argmax(labels)==np.argmax(outputs):
                        train_correct+=1
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_acc=round((train_correct/len(train_set))*100,3)
            print("Training Accuracy & Loss for epoch "+str(e+1)+" : "+str(train_acc)+"% ,"+str(round(train_loss/num_batches_train,3)))
            f.write("Training Accuracy & Loss for epoch "+str(e+1)+" : "+str(train_acc)+"% ,"+str(round(train_loss/num_batches_train,3))+"\n")
            for i,data in enumerate(valloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                labels=torch.squeeze(labels)
                if labels.dim()==0:
                    loss=criterion(outputs,labels.unsqueeze(0))
                else:
                    loss = criterion(outputs,labels)
                labels=labels.detach().cpu().numpy()
                outputs=outputs.detach().cpu().numpy()
                try:
                    res= np.argmax(outputs,axis=1)
                    res=res.reshape(-1,1)
                    labels=labels.reshape(-1,1)
                    res=labels==res
                    for i in res:
                        if i:
                            val_correct+=1
                except:
                    if np.argmax(labels)==np.argmax(outputs):
                        val_correct+=1
                val_loss += loss.item()
            cur_val_acc = round((val_correct/len(val_set))*100,3)
            cur_val_loss = round(val_loss/num_batches_val,3) 
            print("Validation Accuracy & Loss for epoch "+str(e+1)+" : "+str(cur_val_acc)+"% ,"+str(cur_val_loss))
            f.write("Validation Accuracy & Loss for epoch "+str(e+1)+" : "+str(cur_val_acc)+"% ,"+str(cur_val_loss)+"\n")
            if args.early_stop:
                if early_stop(max_val_acc,cur_val_acc,min_val_loss,cur_val_loss):
                    time.sleep(5)
                    sys.exit(0)
                else:
                    if cur_val_acc > max_val_acc:
                        max_val_acc = cur_val_acc
                    if cur_val_loss < min_val_loss:
                        min_val_loss = cur_val_loss
            if args.imagenet:
                torch.save(net,model_name+"_imagenet.pth")
            else:
                torch.save(net,model_name+"_cifar10.pth")
    if args.imagenet:
        torch.save(net,model_name+"_imagenet.pth")
    else:
        torch.save(net,model_name+"_cifar10.pth")
            
def run_code_for_testing(net,model_name):
    transforms = tf.Compose([tf.ToTensor(),tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    if args.imagenet:
        args.resize=256
        assert args.imagenet_test, "ImageNet Test set path not specified"
        print("Loading Test ImageNet Test Dataset............")
        testset = ImageNetClass(args.imagenet_test,transforms=transforms,resize=(args.resize,args.resize))
    else:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transforms)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=True, num_workers=2)
    
    num_batches_test=int(len(testset)/1)

    print("Test Set Size :"+str(len(testset)))
    
    net = net.to(device)
    summary(net,input_size=(3,args.resize,args.resize))
    criterion = nn.CrossEntropyLoss()
    if args.imagenet:
            result_file=model_name+"_testing_imagenet.txt"
    else:
            result_file=model_name+"_testing_cifar10.txt"
    with open(result_file,"w") as f:
            test_loss = 0
            test_correct = 0
            for i,data in enumerate(testloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                labels=torch.squeeze(labels)
                if labels.dim()==0:
                    loss=criterion(outputs,labels.unsqueeze(0))
                else:
                    loss = criterion(outputs,labels)
                labels=labels.detach().cpu().numpy()
                outputs=outputs.detach().cpu().numpy()
                try:
                    res= np.argmax(outputs,axis=1)
                    res=res.reshape(-1,1)
                    labels=labels.reshape(-1,1)
                    res=labels==res
                    for i in res:
                        if i:
                            test_correct+=1
                except:
                    if np.argmax(labels)==np.argmax(outputs):
                        test_correct+=1
                test_loss += loss.item()
            print("Testing Accuracy & Loss : "+str(round((test_correct/len(testset))*100,3))+"% ,"+str(round(test_loss/num_batches_test,3)))
            f.write("Testing Accuracy & Loss : "+str(round((test_correct/len(testset))*100,3))+"% ,"+str(round(test_loss/num_batches_test,3))+"\n")


def which_resnet(name):
    if name=="resnet18":
        net=models.resnet18(pretrained=True)
    elif name=="resnet34":
        net=models.resnet34(pretrained=True)
    elif name=="resnet50":
        net=models.resnet50(pretrained=True)
    elif name=="resnet101":
        net=models.resnet101(pretrained=True)
    elif name=="resnet152":
        net=models.resnet152(pretrained=True)
    elif name=="resnext50":
        net=models.resnext50_32x4d(pretrained=True)
    elif name=="resnext101":
        net=models.resnext101_32x8d(pretrained=True)
    elif name=="wide_resnet50":
        net=models.wide_resnet50_2(pretrained=True)
    elif name=="wide_resnet101":
        net=models.wide_resnet101_2(pretrained=True)
    if name=="resnet18" or name=="resnet34":
        if args.imagenet:
            net.fc=nn.Linear(512,5,bias=True)
        else:    
            net.fc=nn.Linear(512,10,bias=True)
    else:
        if args.imagenet:
            net.fc=nn.Linear(2048,512,bias=True)
            net=nn.Sequential(net,
                                nn.Linear(512,5,bias=True)
                       )
        else:
            net.fc=nn.Linear(2048,512,bias=True)
            net=nn.Sequential(net,
                                nn.Linear(512,10,bias=True)
                       )
    return net

if __name__=="__main__":
    if args.run:
        assert (args.name and args.model), "Model Name or Model Path not specified, please specify both" 
        net = torch.load(args.model)
        run_code_for_testing(net,args.name)
    else:
        assert args.name, "Model Name name specified, please specify as it is used for saving results and the model weights"
        if args.torch_resnet:
            net = which_resnet(args.torch_resnet)
        else:
            net=SkipConnections.BMEnet()
        run_code_for_training(net,args.name)

