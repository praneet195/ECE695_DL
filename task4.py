'''
Please run this code as follows: python3 hw05_task4.py --train_tars <train_tar1.gz>,<train_tar2.gz>,...,<train_tarN.gz> --test_tars <test_tar1.gz>,<test_tar2.gz>,...<test_tarN.gz> --smooth <smooth_flag> --logic <logic_flag>
Example:
python3 hw05_task4.py --train_tars PurdueShapes5-10000-train-noise-50.gz --test_tars Examples/data/PurdueShapes5-1000-test-noise-50.gz --smooth 1 --logic 1 
or
python3 hw05_task4.py --train_tars PurdueShapes5-10000-train-noise-50.gz,PurdueShapes5-10000-train.gz --test_tars Examples/data/PurdueShapes5-1000-test-noise-50.gz,PurdueShapes5-1000-test.gz --smooth 1 --logic 1.
If you want to use all the PurdueShapes5 sets, then ensure you separate them with commas.
The smooth flag incorporates the learning Gaussian Kernel. The logic flag incorporates the Denoising CNN logic. If both are set to 0, you will get the baseline LoadNet2.

Flow of the code:
    The network follows the same structure as LoadNet2. However, 2 new parts have been added to it. 1, is a  learning Gaussian kernel that helps smooth inputs by determining the standard deviation value based on noise in the image. I've used a PyTorch implementation of Gaussian Kernel and have passed sigma as a learnable parameter. 2, a U-net based Denoising CNN as seen here https://github.com/lychengr3x/Image-Denoising-with-Deep-CNNs. Both these help improve the performance of LoadNet2 when the entire PurdueShapes5 Dataset is considered ( original + noise20 + noise50 + noise80) and also only when a single noisy dataset is considered. A new regression path has been added to help the network learning this sigma value as well.

When we use the all datasets, we see that baseline LoadNet gives an accuracy of 72%. However, when the learning kernel and UDCNN is used, we get an 87% overall accuracy considering all testsets at once.
When we only consider the noise-50 dataset, baseline LoadNet gives a 62% accuracy but here we get 74%.

The training and testing code is based on Professor Avi Kak's DLStudio.py. '''
import math
import numbers
import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary
import numpy as np
from PIL import ImageFilter
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import argparse


parser= argparse.ArgumentParser(description="Helps navigate this code better")
parser.add_argument("--train_tars",type=str,default=None,help=" Specify which tar to use for training")
parser.add_argument("--test_tars",type=str,default=None,help = " Specify which tar to use for testing ")
parser.add_argument("--smooth",type=int,default=None,help = " Specify which tar to use for testing ")
parser.add_argument("--logic",type=int,default=None,help = " Specify which tar to use for testing ")
args = parser.parse_args()


class DetectAndLocalize(nn.Module):
        def __init__(self, epochs = 10, dataset_file=None, save_models_root="./",batch_size =8 ,lr=1e-5, momentum=0.9, device=  torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),debug_train=True,debug_test=True, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test
            self.epochs = epochs
            self.momentum = momentum
            self.batch_size = batch_size
            self.learning_rate = lr
            self.debug_train = debug_train
            self.debug_test = debug_test
            self.device = device
            self.save_models_root = save_models_root
            self.dataset_file = dataset_file

        class PurdueShapes5Dataset(torch.utils.data.Dataset):
            def __init__(self,dataroot,train_or_test,dataset_file,transform=None):
                super(DetectAndLocalize.PurdueShapes5Dataset,self).__init__()
                self.dataroot=dataroot
                self.dataset_files = dataset_file.split(",")
                self.concat=[]
                self.smooth = args.smooth
                for dataset_file in self.dataset_files:
                    if train_or_test == 'train':
                        print(dataset_file)
                        dataset_pt = dataset_file[:-2]+"pt"
                        label_pt = "torch-saved-PurdueShapes5-label-map.pt"
                        if os.path.exists(dataset_pt) and  os.path.exists(label_pt):
                            print("\n Loading training data from torch-saved-archive")
                            self.dataset = torch.load(dataset_pt)
                            if self.smooth:
                                self.add_noise_labels(dataset_file)
                            self.label_map = torch.load(label_pt)
                            self.class_labels = dict(map(reversed, self.label_map.items()))
                            self.transform = transform
                        else:
                            print("\n First time loading...")
                            root_dir = self.dataroot
                            f = gzip.open(root_dir+dataset_file,"rb")
                            dataset = f.read()
                            if sys.version_info[0] == 3:
                                self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                            else:
                                self.dataset, self.label_map = pickle.loads(dataset)
                            torch.save(self.dataset, dataset_pt)
                            torch.save(self.label_map, label_pt)
                            if self.smooth:
                                self.add_noise_labels(dataset_file)
                            self.class_labels = dict(map(reversed, self.label_map.items()))
                            self.transform = transform
                    else:
                        root_dir = self.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
                        if self.smooth:
                            self.add_noise_labels(dataset_file)
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                    self.concat.append(self.dataset)
                self.dataset = torch.utils.data.ConcatDataset(self.concat) 

            def add_noise_labels(self,dataset_file):
                if "noise-20" in dataset_file:
                    for i in self.dataset.keys():
                        self.dataset[i].append(0.2)
                elif "noise-50" in dataset_file:
                    for i in self.dataset.keys():
                        self.dataset[i].append(0.5)
                elif "noise-80" in dataset_file:
                    for i in self.dataset.keys():
                        self.dataset[i].append(0.9)
                else:
                    for i in self.dataset.keys():
                        self.dataset[i].append(0.1)
                    

            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self,idx):
                r = np.array(self.dataset[idx][0])
                g = np.array( self.dataset[idx][1])
                b = np.array(self.dataset[idx][2])
                R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
                im_tensor = torch.zeros(3,32,32, dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
                bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
                if self.smooth:
                    noise_tensor = torch.tensor(self.dataset[idx][5]) 
                    sample = {'image' : im_tensor, 
                          'bbox' : bb_tensor,
                          'label' : self.dataset[idx][4] ,
                          'noise' : self.dataset[idx][5]}
                else:
                    sample = {'image' : im_tensor, 
                          'bbox' : bb_tensor,
                          'label' : self.dataset[idx][4] 
                          }
                if self.transform:
                     sample = self.transform(sample)
                return sample

        def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test ):       
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                               batch_size=self.batch_size,shuffle=True, num_workers=4)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.batch_size,shuffle=False, num_workers=4)


        class SkipBlock(nn.Module):
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DetectAndLocalize.SkipBlock, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                norm_layer = nn.BatchNorm2d
                self.bn = norm_layer(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

            def forward(self, x):
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


################## I got this from the PyTorch website ####################
        class GaussianSmoothing(nn.Module):
            def __init__(self, channels, kernel_size, sigma, dim=2):
                super(DetectAndLocalize.GaussianSmoothing, self).__init__()
                if isinstance(kernel_size, numbers.Number):
                    kernel_size = [kernel_size] * dim
                if isinstance(sigma, numbers.Number):
                    sigma = [sigma] * dim

            # The gaussian kernel is the product of the
            # gaussian function of each dimension.
                kernel = 1
                meshgrids = torch.meshgrid(
                    [
                        torch.arange(size, dtype=torch.float32)
                        for size in kernel_size
                    ]
                )
                for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                    mean = (size - 1) / 2
                    kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                          torch.exp(-((mgrid - mean) / std) ** 2 / 2)

            # Make sure sum of values in gaussian kernel equals 1.
                kernel = kernel / torch.sum(kernel)

            # Reshape to depthwise convolutional weight
                kernel = kernel.view(1, 1, *kernel.size())
                kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

                self.register_buffer('weight', kernel)
                self.groups = channels

                self.conv = F.conv2d

            def forward(self, input):
                return self.conv(input, weight=self.weight, groups=self.groups)


        class LOADnet2(nn.Module):
            """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            LOADnet2 uses both convo and linear layers for regression
            """ 
            def __init__(self, skip_connections=True, depth=32,smooth=args.smooth):
                super(DetectAndLocalize.LOADnet2, self).__init__()
                self.smooth=smooth
                self.logic=args.logic
################ This is the UDCNN #############################
                if self.logic:
                        self.conv_l = nn.ModuleList()
                        self.conv_l.append(nn.Conv2d(3, 64, 3, padding=1))
                        self.conv_l.extend([nn.Conv2d(64, 64, 3, padding=1) for _ in range(16)])
                        self.conv_l.append(nn.Conv2d(64, 3, 3, padding=1))
                        for i in range(len(self.conv_l[:-1])):
                            nn.init.kaiming_normal_(self.conv_l[i].weight.data, nonlinearity='relu')
                        self.bn_l = nn.ModuleList()
                        self.bn_l.extend([nn.BatchNorm2d(64, 64) for _ in range(16)])
                        for i in range(16):
                            nn.init.constant_(self.bn_l[i].weight.data, 1.25 * np.sqrt(64))
##################### This is the learnable Gaussian Kernel ##############################################
                if self.smooth:
                    self.sigma = torch.FloatTensor(torch.nn.Parameter(torch.rand(1)))
                    #print(self.sigma.requires_grad)
                    self.smoothing = DetectAndLocalize.GaussianSmoothing(3, 3, self.sigma)

                self.pool_count = 3
                self.depth = depth // 2
                self.conv = nn.Conv2d(3, 64, 3, padding=1)

                self.pool = nn.MaxPool2d(2, 2)
                self.skip64 = torch.nn.ModuleList([DetectAndLocalize.SkipBlock(64,64,skip_connections=skip_connections) for i in range(self.depth //4)])
                self.skip64ds = DetectAndLocalize.SkipBlock(64, 64, 
                                           downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DetectAndLocalize.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128 = torch.nn.ModuleList([DetectAndLocalize.SkipBlock(128, 128,  skip_connections=skip_connections) for i in range(self.depth//4)])
                self.skip128ds = DetectAndLocalize.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
                self.fc2 =  nn.Linear(1000, 5)
                ##  for regression
                self.regr_seqn = nn.Sequential(
                nn.Conv2d(64,128,3,padding=1,stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128,256,3,padding=1,stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,512,3,padding=1,stride=1),
                nn.ReLU(inplace=True),
                )
                self.regr_seqn2=nn.Sequential(
                nn.Linear(131072,2048),
                nn.Dropout(p=0.5,inplace=True),
                nn.Linear(2048,512),
                nn.Dropout(p=0.5,inplace=True),
                nn.Linear(512,4)
                )
                
                if self.smooth:
                    self.noise_seqn = nn.Sequential(
                nn.Conv2d(64,128,3,padding=1,stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128,256,3,padding=1,stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,512,3,padding=1,stride=1),
                nn.ReLU(inplace=True),
                )
                    self.noise_seqn2=nn.Sequential(
                nn.Linear(131072,2048),
                nn.Dropout(p=0.5,inplace=True),
                nn.Linear(2048,512),
                nn.Dropout(p=0.5,inplace=True),
                nn.Linear(512,1)
                )

            def forward(self, x):
                if self.smooth:
                    x = self.smoothing(x)
                    x = F.pad(x, (1, 1, 1, 1), mode='reflect')
                if self.logic:
                    h = F.relu(self.conv_l[0](x))
                    for i in range(16):
                        h = F.relu(self.bn_l[i](self.conv_l[i+1](h)))
                    y = self.conv_l[17](h) + x
                    y = self.conv_l[17](h) + x
                    x = self.pool(torch.nn.functional.relu(self.conv(y)))
                else:
                    x = self.pool(torch.nn.functional.relu(self.conv(x)))
                ## The labeling section:
                x1 = x.clone()
                for i in self.skip64:
                    x1 = i(x1)
                x1 = self.skip64ds(x1)
                for i in self.skip64:
                    x1 = i(x1)                                           
                x1 = self.skip64to128(x1)
                for i in self.skip128:
                    x1 = i(x1)                                               
                x1 = self.skip128ds(x1)                                             
                for i in self.skip128:
                    x1 = i(x1)                                               
                x1 = x1.view(-1, 128 * (32 // 2**self.pool_count)**2 )
                x1 = torch.nn.functional.relu(self.fc1(x1))
                x1 = self.fc2(x1)
                ## The Bounding Box regression:
                x2 = x.clone()
                x2 = self.regr_seqn(x)
                # flatten
                x2 = x2.view(x.size(0), -1)
                x2 = self.regr_seqn2(x2)
                
                if self.smooth:
                    x3 = self.noise_seqn(x)
                # flatten
                    x3 = x3.view(x.size(0), -1)
                    x3 = self.noise_seqn2(x3)

                    return x1,x2,x3
                else:
                    return x1,x2

        def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):        
            self.smooth=args.smooth
            self.path_saved_model=self.dataset_file[:-3]+"_original.pth"
            filename_for_out1 = "./logs/original_performance_numbers_" + str(self.epochs) + "_label_"+self.dataset_file[:-3]+".txt"
            filename_for_out2 = "./logs/original_performance_numbers_" + str(self.epochs) + "_regres_"+self.dataset_file[:-3]+".txt"
            FILE1 = open(filename_for_out1, 'w')
            FILE2 = open(filename_for_out2, 'w')
            net = type(net)()
            net.load_state_dict(net.state_dict())
            net = net.to(self.device)
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.MSELoss()
            criterion3 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.learning_rate,momentum=self.momentum)
            for epoch in range(self.epochs):  
                running_loss_labeling = 0.0
                running_loss_regression = 0.0       
                if self.smooth:
                    running_noise =0.0
                for i, data in enumerate(self.train_dataloader):
                    gt_too_small = False
                    if self.smooth:
                        inputs, bbox_gt, labels, noise_gt = data['image'], data['bbox'], data['label'], data["noise"]
                    else:
                        inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
                    if self.debug_train and i % 500 == 499:
                        print("\n\n[epoch=%d iter=%d:] Ground Truth:     " % (epoch+1, i+1) + 
                        ' '.join('%10s' % self.dataserver_train.class_labels[labels[j].item()] for j in range(self.batch_size)))
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    bbox_gt = bbox_gt.to(self.device)
                    if self.smooth:
                        noise_gt = noise_gt.float()
                        noise_gt = noise_gt.to(self.device)
                        noise_gt= noise_gt.unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    outputs_label = outputs[0]
                    bbox_pred = outputs[1]
                    if self.smooth:
                        noise_pred = outputs[2]
                    if self.debug_train and i % 500 == 499:
                        inputs_copy = inputs.detach().clone()
                        inputs_copy = inputs_copy.cpu()
                        bbox_pc = bbox_pred.detach().clone()
                        bbox_pc[bbox_pc<0] = 0
                        bbox_pc[bbox_pc>31] = 31
                        bbox_pc[torch.isnan(bbox_pc)] = 0
                        _, predicted = torch.max(outputs_label.data, 1)
                        print("[epoch=%d iter=%d:] Predicted Labels: " % (epoch+1, i+1) + 
                         ' '.join('%10s' % self.dataserver_train.class_labels[predicted[j].item()] 
                                           for j in range(self.batch_size)))
                        for idx in range(self.batch_size):
                            i1 = int(bbox_gt[idx][1])
                            i2 = int(bbox_gt[idx][3])
                            j1 = int(bbox_gt[idx][0])
                            j2 = int(bbox_gt[idx][2])
                            k1 = int(bbox_pc[idx][1])
                            k2 = int(bbox_pc[idx][3])
                            l1 = int(bbox_pc[idx][0])
                            l2 = int(bbox_pc[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            inputs_copy[idx,0,i1:i2,j1] = 255
                            inputs_copy[idx,0,i1:i2,j2] = 255
                            inputs_copy[idx,0,i1,j1:j2] = 255
                            inputs_copy[idx,0,i2,j1:j2] = 255
                            inputs_copy[idx,2,k1:k2,l1] = 255                      
                            inputs_copy[idx,2,k1:k2,l2] = 255
                            inputs_copy[idx,2,k1,l1:l2] = 255
                            inputs_copy[idx,2,k2,l1:l2] = 255
                    loss_labeling = criterion1(outputs_label, labels)
                    loss_labeling.backward(retain_graph=True)        
                    loss_regression = criterion2(bbox_pred, bbox_gt)
                    loss_regression.backward(retain_graph=True)
                    if self.smooth:
                        loss_noise = criterion3(noise_pred,noise_gt)
                        loss_noise.backward(retain_graph=True)
                    optimizer.step()
                    running_loss_labeling += loss_labeling.item()    
                    running_loss_regression += loss_regression.item()       
                    if self.smooth:
                        running_noise += loss_noise.item()
                    if i % 500 == 499:    
                        avg_loss_labeling = running_loss_labeling / float(500)
                        avg_loss_regression = running_loss_regression / float(500)
                        if self.smooth:
                            avg_noise = running_noise / float(500)
                            print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f loss_noise: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression,avg_noise))
                        else:
                            print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                        FILE1.write("%.3f\n" % avg_loss_labeling)
                        FILE1.flush()
                        FILE2.write("%.3f\n" % avg_loss_regression)
                        FILE2.flush()
                        running_loss_labeling = 0.0
                        running_loss_regression = 0.0
                        if self.smooth:
                            running_noise=0.0
                        

            print("\nFinished Training\n")
            self.save_model(net)

        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.path_saved_model)

        def run_code_for_testing_detection_and_localization(self, net):
            self.smooth = args.smooth
            self.path_saved_model=self.dataset_file[:-3]+"_original.pth"
            net.load_state_dict(torch.load(self.path_saved_model))
            correct = 0
            total = 0
            confusion_matrix = torch.zeros(len(self.dataserver_train.class_labels), 
                                           len(self.dataserver_train.class_labels))
            class_correct = [0] * len(self.dataserver_train.class_labels)
            class_total = [0] * len(self.dataserver_train.class_labels)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    if self.smooth:
                        images, bounding_box, labels, noise = data['image'], data['bbox'], data['label'], data["noise"]
                    else:
                        images, bounding_box, labels = data['image'], data['bbox'], data['label']
                    labels = labels.tolist()
                    if self.debug_test and i % 50 == 0:
                        print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%10s' % 
    self.dataserver_train.class_labels[labels[j]] for j in range(self.batch_size)))
                    outputs = net(images)
                    outputs_label = outputs[0]
                    outputs_regression = outputs[1]
                    outputs_regression[outputs_regression < 0] = 0
                    outputs_regression[outputs_regression > 31] = 31
                    outputs_regression[torch.isnan(outputs_regression)] = 0
                    output_bb = outputs_regression.tolist()
                    if self.smooth:
                        output_noise = outputs[2]
                    _, predicted = torch.max(outputs_label.data, 1)
                    predicted = predicted.tolist()
                    if self.debug_test and i % 50 == 0:
                        print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
 self.dataserver_train.class_labels[predicted[j]] for j in range(self.batch_size)))
                        if self.smooth:
                            print("Predicted Sigma : "+str(output_noise.tolist()) +" Actual Sigma : "+str(noise.tolist()))
                        for idx in range(self.batch_size):
                            i1 = int(bounding_box[idx][1])
                            i2 = int(bounding_box[idx][3])
                            j1 = int(bounding_box[idx][0])
                            j2 = int(bounding_box[idx][2])
                            k1 = int(output_bb[idx][1])
                            k2 = int(output_bb[idx][3])
                            l1 = int(output_bb[idx][0])
                            l2 = int(output_bb[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            images[idx,0,i1:i2,j1] = 255
                            images[idx,0,i1:i2,j2] = 255
                            images[idx,0,i1,j1:j2] = 255
                            images[idx,0,i2,j1:j2] = 255
                            images[idx,2,k1:k2,l1] = 255                      
                            images[idx,2,k1:k2,l2] = 255
                            images[idx,2,k1,l1:l2] = 255
                            images[idx,2,k2,l1:l2] = 255
                    for label,prediction in zip(labels,predicted):
                        confusion_matrix[label][prediction] += 1
                    total += len(labels)
                    correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
                    comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
                    for j in range(self.batch_size):
                        label = labels[j]
                        class_correct[label] += comp[j]
                        class_total[label] += 1
            print("\n")
            for j in range(len(self.dataserver_train.class_labels)):
                print('Prediction accuracy for %5s : %2d %%' % (
              self.dataserver_train.class_labels[j], 100 * class_correct[j] / class_total[j]))
            print("\n\n\nOverall accuracy of the network on the 1000 test images: %d %%" % 
                                                                   (100 * correct / float(total)))
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                "
            for j in range(len(self.dataserver_train.class_labels)):  
                                 out_str +=  "%15s" % self.dataserver_train.class_labels[j]   
            print(out_str + "\n")
            for i,label in enumerate(self.dataserver_train.class_labels):
                out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                 for j in range(len(self.dataserver_train.class_labels))]
                out_percents = ["%.2f" % item.item() for item in out_percents]
                out_str = "%12s:  " % self.dataserver_train.class_labels[i]
                for j in range(len(self.dataserver_train.class_labels)): 
                                                       out_str +=  "%15s" % out_percents[j]
                print(out_str)






if __name__=="__main__":
   # assert args.train_tar, args.test_tar, "Please Specify train and test tar"
    detector = DetectAndLocalize( )
    dataserver_train = DetectAndLocalize.PurdueShapes5Dataset(
                                       train_or_test = 'train',
                                       dataroot = './',
    #                                   dataset_file = "PurdueShapes5-20-train.gz",
                                       dataset_file = args.train_tars
                                                                          )
    dataserver_test = DetectAndLocalize.PurdueShapes5Dataset(
                                       train_or_test = 'test',
                                       dataroot='./',
    #                                   dataset_file = "PurdueShapes5-20-test.gz"
                                       dataset_file = args.test_tars
                                                                      )
    detector.dataserver_train = dataserver_train
    detector.dataserver_test = dataserver_test
    
    detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)
    
    model = detector.LOADnet2(skip_connections=True, depth=32)
     
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_layers = len(list(model.parameters())) 
    print("Number of trainable Parameters :"+str(model_params))
    print("Number of layers :"+str(model_layers))
    #dls.show_network_summary(model)
    detector.dataset_file = "this_is_task4_4_with_everything_inb_it"
    detector.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model)
    #detector.run_code_for_training_with_CrossEntropy_and_BCE_Losses(model)
    detector.run_code_for_testing_detection_and_localization(model)
    

