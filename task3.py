'''
Please run this code as follows: python3 hw05_task3.py --train_tars <train_tar1.gz>,<train_tar2.gz>,<train_tar3.gz>,<train_tar4.gz> --test_tars <test_tar1.gz>,<test_tar2.gz>,<test_tar3.gz>,<test_tar4.gz> --smooth 1
This Code contains a Noise Classifier that helps classify images based on the noise levels in them. The model is based on VGG but I've incorporated dropout and batch_norm into it. I call it NoiseNet. It achieves a test accuracy of 96%. The ground truth noise levels were determined from dataset file names as follows:
    0: "No Noise"
    1: "Noise-20"
    2: "Noise-50"
    3: "Noise-80"

Use smooth parameter as 1 to use noise classifier and then smooth images based on its output or else it'll be the default inputs to network from all datasets available. In the next task, I've created a network that does it all.

This is a continuation of task 3. This uses a pre-trained noise classifier to help determine the level of Gaussian smoothing to be applied to different noise images. Another way of doing this would be to only run testing by selecting weights obtained from task2. If you want the weights you canget them as well. In this case, I have trained loadnet similar to the best results in task 2. Please ask me for the pre-trained weights if this needs to be run.You can mail me at singh671@purdue.edu

This Code is based on Professor Avi Kak's DLStudio.py. I do not own this code. Very similar code format to my previous task. In this task, I've used the imgaug library to apply various smoothing filters to the noisy datasets based on the output of my NoiseNet. Although, computationally slow this achieves a 5-6% better accuracy when baseline LoadNetv2 is used. However, this entire network is jointly optimized in my next task to elarn everyhting together at once.
Also at the end of this file, I've attached the file used to train the NoiseNet().

'''
import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary
import numpy as np
from PIL import ImageFilter, Image
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import argparse
import imgaug as ia
import imgaug.augmenters as iaa

parser= argparse.ArgumentParser(description="Helps navigate this code better")
parser.add_argument("--train_tars",type=str,default=None,help=" Specify which tar to use for training")
parser.add_argument("--test_tars",type=str,default=None,help = " Specify which tar to use for testing ")
parser.add_argument("--smooth",type=int,default=0,help=" Use smoothing based on noise classifier ")
args = parser.parse_args()


class DetectAndLocalize(nn.Module):
        def __init__(self, epochs = 8, dataset_file=None, save_models_root="./",batch_size =8 ,lr=1e-5, momentum=0.9, device=  torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'),debug_train=True,debug_test=True, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
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

        
        class NoiseNet(nn.Module):
            ''' Similar to VGG structure '''
            def __init__(self):
                super(DetectAndLocalize.NoiseNet,self).__init__()
                self.bn1 = nn.BatchNorm2d(3)
                self.conv1 = nn.Conv2d(3,64,3,padding=1,stride=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv2 = nn.Conv2d(64,64,3,padding=1,stride=1)
                self.bn3 = nn.BatchNorm2d(64)
                self.max_pool1 = nn.MaxPool2d(2,stride=2)
                self.conv3 = nn.Conv2d(64,128,3,padding=1,stride=1)
                self.bn4 = nn.BatchNorm2d(128)
                self.conv4 = nn.Conv2d(128,128,3,padding=1,stride=1)
                self.bn5 = nn.BatchNorm2d(128)
                self.max_pool2 = nn.MaxPool2d(2,stride=2)
                self.conv5 = nn.Conv2d(128,256,3,padding=1,stride=1)
                self.bn6 = nn.BatchNorm2d(256)
                self.conv6 = nn.Conv2d(256,256,3,padding=1,stride=1)
                self.bn7 = nn.BatchNorm2d(256)
                self.conv7 = nn.Conv2d(256,256,3,padding=1,stride=1)
                self.bn8 = nn.BatchNorm2d(256)
                self.max_pool3 = nn.MaxPool2d(2,stride=2)
                self.conv8 = nn.Conv2d(256,512,3,padding=1,stride=1)
                self.bn9 = nn.BatchNorm2d(512)
                self.conv9 = nn.Conv2d(512,512,3,padding=1,stride=1)
                self.bn10 = nn.BatchNorm2d(512)
                self.conv10 = nn.Conv2d(512,512,3,padding=1,stride=1)
                self.bn11 = nn.BatchNorm2d(512)
                self.max_pool4 = nn.MaxPool2d(2,stride=2)
                self.fc1 = nn.Linear(2048,512)
                self.dropout = nn.Dropout(p=0.5)
                self.fc2 = nn.Linear(512,4)

            def forward(self,x):
                x = self.max_pool1(F.relu(self.bn3(self.conv2(F.relu(self.bn2(self.conv1(self.bn1(x))))))))
                x = self.max_pool2(F.relu(self.bn5(self.conv4(F.relu(self.bn4(self.conv3(x)))))))
                x = self.max_pool3(F.relu(self.bn8(self.conv7(F.relu(self.bn7(self.conv6(F.relu(self.bn6(self.conv5(x))))))))))
                x = self.max_pool4(F.relu(self.bn11(self.conv10(F.relu(self.bn10(self.conv9(F.relu(self.bn9(self.conv8(x))))))))))
                x = x.view(-1,2048)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        
        class PurdueShapes5Dataset(torch.utils.data.Dataset):
            def __init__(self,dataroot,train_or_test,dataset_files,transform=None,smooth=args.smooth):
                super(DetectAndLocalize.PurdueShapes5Dataset,self).__init__()
                self.dataroot=dataroot
                self.concat_datasets=[]
                self.datasets = dataset_files.split(",")
                self.smooth = smooth
                sorted(self.datasets)
                if train_or_test == 'train':
                    for i,dataset_file in enumerate(self.datasets):
                        print(i,dataset_file)
                        dataset_pt = dataset_file[:-2]+"pt"
                        label_pt = "torch-saved-PurdueShapes5-label-map_alldatasets.pt"
                        if os.path.exists(dataset_pt) and  os.path.exists(label_pt):
                            print("\n Loading training data from torch-saved-archive")
                            self.dataset = torch.load(dataset_pt)
                            self.label_map = torch.load(label_pt)
                            self.class_labels = dict(map(reversed, self.label_map.items()))
                            self.transform = transform
                            if self.smooth:
                                self.load_and_smooth_images()
                            self.concat_datasets.append(self.dataset)
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
                            self.class_labels = dict(map(reversed, self.label_map.items()))
                            self.transform = transform
                            if self.smooth:
                                self.load_and_smooth_images()
                            self.concat_datasets.append(self.dataset)
                else:
                    for i,dataset_file in enumerate(self.datasets):
                        print(i,dataset_file)
                        root_dir = self.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
                        if self.smooth:
                            self.load_and_smooth_images()
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                        self.concat_datasets.append(self.dataset)
                self.dataset = torch.utils.data.ConcatDataset(self.concat_datasets)
                print("All datasets appended. Final dataset length :"+str(len(self.dataset)))

            
            def load_and_smooth_images(self):
                print("Applying Smoothing using Noise Classifier Net with value ")
                noisenet = DetectAndLocalize.NoiseNet()
                self.path_saved_model="praneet_noise.pth"
                noisenet.load_state_dict(torch.load(self.path_saved_model))
                count=0
                for i in range(len(self.dataset)):
                        r = np.array(self.dataset[i][0])
                        g = np.array( self.dataset[i][1])
                        b = np.array(self.dataset[i][2])
                        bbox=self.dataset[i][3]
                        label=self.dataset[i][4]
                        R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
                        im_tensor = torch.zeros(3,32,32, dtype=torch.float)
                        im_tensor[0,:,:] = torch.from_numpy(R)
                        im_tensor[1,:,:] = torch.from_numpy(G)
                        im_tensor[2,:,:] = torch.from_numpy(B)
                        im_tensor = im_tensor.view(1,3,32,32)
                        outputs = noisenet(im_tensor)
                        outputs_label = outputs
                        _, predicted = torch.max(outputs_label.data, 1)
                        predicted = predicted.tolist()
                        im_npy = im_tensor.view(1,32,32,3).detach().numpy()
                        if predicted[0] == 1:
                            seq=iaa.Sequential([iaa.GaussianBlur(sigma=0.3)])
                            img_aug = seq(images=im_npy)
                        elif predicted[0] == 2:
                            seq=iaa.Sequential([iaa.GaussianBlur(sigma=0.9)])
                            img_aug = seq(images=im_npy)
                        elif predicted[0] == 3:
                            seq=iaa.Sequential([iaa.BilateralBlur(d=int(9))])
                            img_aug = seq(images=im_npy)
                        else:
                            img_aug = im_npy
                        img_aug = img_aug.reshape(3,32,32)
                        self.dataset[count]=[img_aug[0],img_aug[1],img_aug[2],bbox,label]
                        count+=1


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
                sample = {'image' : im_tensor,
                          'bbox' : bb_tensor,
                          'label' : self.dataset[idx][4] }
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


        class LOADnet2(nn.Module):
            """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            LOADnet2 uses both convo and linear layers for regression
            """ 
            def __init__(self, skip_connections=True, depth=32):
                super(DetectAndLocalize.LOADnet2, self).__init__()
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
                self.conv_seqn = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.fc_seqn = nn.Sequential(
                    nn.Linear(16384, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 4)
                )

            def forward(self, x):
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
                x2 = self.conv_seqn(x)
                x2 = self.conv_seqn(x2)
                # flatten
                x2 = x2.view(x.size(0), -1)
                x2 = self.fc_seqn(x2)
                return x1,x2


        def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):        
            self.path_saved_model=self.dataset_file[:-3]+".pth"
            filename_for_out1 = "./logs/performance_numbers_" + str(self.epochs) + "_label_"+self.dataset_file+".txt"
            filename_for_out2 = "./logs/performance_numbers_" + str(self.epochs) + "_regres_"+self.dataset_file+".txt"
            FILE1 = open(filename_for_out1, 'w')
            FILE2 = open(filename_for_out2, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.device)
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.learning_rate, momentum=self.momentum)
            for epoch in range(self.epochs):  
                running_loss_labeling = 0.0
                running_loss_regression = 0.0       
                for i, data in enumerate(self.train_dataloader):
                    gt_too_small = False
                    inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
                    if self.debug_train and i % 500 == 499:
                        print("\n\n[epoch=%d iter=%d:] Ground Truth:     " % (epoch+1, i+1) + 
                        ' '.join('%10s' % self.dataserver_train.class_labels[labels[j].item()] for j in range(self.batch_size)))
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    bbox_gt = bbox_gt.to(self.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    outputs_label = outputs[0]
                    bbox_pred = outputs[1]
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
                    loss_regression.backward()
                    optimizer.step()
                    running_loss_labeling += loss_labeling.item()    
                    running_loss_regression += loss_regression.item()                
                    if i % 500 == 499:    
                        avg_loss_labeling = running_loss_labeling / float(500)
                        avg_loss_regression = running_loss_regression / float(500)
                        print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                        FILE1.write("%.3f\n" % avg_loss_labeling)
                        FILE1.flush()
                        FILE2.write("%.3f\n" % avg_loss_regression)
                        FILE2.flush()
                        running_loss_labeling = 0.0
                        running_loss_regression = 0.0


            print("\nFinished Training\n")
            self.save_model(net)

        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.path_saved_model)

        def run_code_for_testing_detection_and_localization(self, net):
            self.path_saved_model=self.dataset_file[-3]+".pth"
            net.load_state_dict(torch.load(self.path_saved_model))
            correct = 0
            total = 0
            confusion_matrix = torch.zeros(len(self.dataserver_train.class_labels), 
                                           len(self.dataserver_train.class_labels))
            class_correct = [0] * len(self.dataserver_train.class_labels)
            class_total = [0] * len(self.dataserver_train.class_labels)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
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
                    _, predicted = torch.max(outputs_label.data, 1)
                    predicted = predicted.tolist()
                    if self.debug_test and i % 50 == 0:
                        print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
 self.dataserver_train.class_labels[predicted[j]] for j in range(self.batch_size)))
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
                                       dataset_files = args.train_tars
                                                                          )
    dataserver_test = DetectAndLocalize.PurdueShapes5Dataset(
                                       train_or_test = 'test',
                                       dataroot='./',
    #                                   dataset_file = "PurdueShapes5-20-test.gz"
                                       dataset_files = args.test_tars
                                                                      )

  #  dataserver_train.load_and_augment_images()
    detector.dataserver_train = dataserver_train
   # dataserver_test.load_and_augment_images()
    detector.dataserver_test = dataserver_test
    detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)
    
    model = detector.LOADnet2(skip_connections=True, depth=32)
     
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_layers = len(list(model.parameters())) 
    print("Number of trainable Parameters :"+str(model_params))
    print("Number of layers :"+str(model_layers))
    #dls.show_network_summary(model)
    detector.dataset_file = "noiseplussmoothing"
    detector.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model)
    #detector.run_code_for_training_with_CrossEntropy_and_BCE_Losses(model)
    detector.run_code_for_testing_detection_and_localization(model)
    
#######################################################################################################################################################################################################################
'''
Please run this code as follows: python3 hw05_task3.py --train_tars <train_tar1.gz>,<train_tar2.gz>,<train_tar3.gz>,<train_tar4.gz> --test_tars <test_tar1.gz>,<test_tar2.gz>,<test_tar3.gz>,<test_tar4.gz>

where I gave train_tar1.gz = Examples/data/PurdueShapes5-10000-train.gz, train_tar2 = Examples/data/PurdueShapes5-10000-noise-20.gz and so on, test_tar1.gz = Examples/data/PurdueShapes5-1000-test.gz, test_tar2.gz = Examples/data/PurdueShapes5-1000-test-noise-20.gz and so on.
Make sure all tars are separated by a comma.

This Code contains a Noise Classifier that helps classify images based on the noise levels in them. The model is based on VGG but I've incorporated dropout and batch_norm into it. I call it NoiseNet. It achieves a test accuracy of 96%. The ground truth noise levels were determined from dataset file names as follows:
    0: "No Noise"
    1: "Noise-20"
    2: "Noise-50"
    3: "Noise-80"

The Code for training and testing has been taken from Prof. Kak's DLStudio with slight modifications on my side.
    
import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary
import numpy as np
from PIL import ImageFilter, Image
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import argparse
import imgaug as ia
import imgaug.augmenters as iaa

parser= argparse.ArgumentParser(description="Helps navigate this code better")
parser.add_argument("--train_tars",type=str,default=None,help=" Specify which tar to use for training")
parser.add_argument("--test_tars",type=str,default=None,help = " Specify which tar to use for testing ")
args = parser.parse_args()


class NoiseClassifier(nn.Module):
        def __init__(self, epochs = 15, dataset_file=None, save_models_root="./",batch_size =8 ,lr=1e-3, momentum=0.9, device=  torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'),debug_train=True,debug_test=True, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
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
            self.dataset_file = "all_datasets"

        class PurdueShapes5Dataset(torch.utils.data.Dataset):
            def __init__(self,dataroot,train_or_test,dataset_files,transform=None):
                super(NoiseClassifier.PurdueShapes5Dataset,self).__init__()
                self.dataroot=dataroot
                self.concat_datasets=[]
                self.datasets = dataset_files.split(",")
                sorted(self.datasets)
                print(self.datasets)
                if train_or_test == 'train':
                    for i,dataset_file in enumerate(self.datasets):
                        print(i,dataset_file)
                        dataset_pt = dataset_file[:-2]+"pt"
                        print(dataset_pt)
                        label_pt = "torch-saved-PurdueShapes5-label-map_noise.pt"
                        if os.path.exists(dataset_pt) and  os.path.exists(label_pt):
                            print("\n Loading training data from torch-saved-archive")
                            self.dataset = torch.load(dataset_pt)
                            self.noise_labels(dataset_file,i)
                            self.concat_datasets.append(self.dataset)
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
                            self.noise_labels(dataset_file,i)
                            self.concat_datasets.append(self.dataset)
                else:
                    for i,dataset_file in enumerate(self.datasets):
                        print(i,dataset_file)
                        root_dir = self.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        self.noise_labels(dataset_file,i)
                        self.concat_datasets.append(self.dataset)
                self.dataset = torch.utils.data.ConcatDataset(self.concat_datasets)
                print("Len :"+str(len(self.dataset)))


            def noise_labels(self,dataset_file,cid):
                if "noise-20" in dataset_file:
                    for k in self.dataset.values():
                        k[4]=cid
                elif "noise-50" in dataset_file:
                    for k in self.dataset.values():
                        k[4]=cid
                elif "noise-80" in dataset_file:
                    for k in self.dataset.values():
                        k[4]=cid
                else:
                    for k in self.dataset.values():
                        k[4]=cid

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
                sample = {'image' : im_tensor,
                          'label' : self.dataset[idx][4] }
                return sample


        def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test ):
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                               batch_size=self.batch_size,shuffle=True, num_workers=4)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.batch_size,shuffle=False, num_workers=4)
            
            
        class NoiseNet(nn.Module):
            # Similar to VGG structure
            def __init__(self):
                super(NoiseClassifier.NoiseNet,self).__init__()
                self.bn1 = nn.BatchNorm2d(3)
                self.conv1 = nn.Conv2d(3,64,3,padding=1,stride=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv2 = nn.Conv2d(64,64,3,padding=1,stride=1)
                self.bn3 = nn.BatchNorm2d(64)
                self.max_pool1 = nn.MaxPool2d(2,stride=2)
                self.conv3 = nn.Conv2d(64,128,3,padding=1,stride=1)
                self.bn4 = nn.BatchNorm2d(128)
                self.conv4 = nn.Conv2d(128,128,3,padding=1,stride=1)
                self.bn5 = nn.BatchNorm2d(128)
                self.max_pool2 = nn.MaxPool2d(2,stride=2)
                self.conv5 = nn.Conv2d(128,256,3,padding=1,stride=1)
                self.bn6 = nn.BatchNorm2d(256)
                self.conv6 = nn.Conv2d(256,256,3,padding=1,stride=1)
                self.bn7 = nn.BatchNorm2d(256)
                self.conv7 = nn.Conv2d(256,256,3,padding=1,stride=1)
                self.bn8 = nn.BatchNorm2d(256)
                self.max_pool3 = nn.MaxPool2d(2,stride=2)
                self.conv8 = nn.Conv2d(256,512,3,padding=1,stride=1)
                self.bn9 = nn.BatchNorm2d(512)
                self.conv9 = nn.Conv2d(512,512,3,padding=1,stride=1)
                self.bn10 = nn.BatchNorm2d(512)
                self.conv10 = nn.Conv2d(512,512,3,padding=1,stride=1)
                self.bn11 = nn.BatchNorm2d(512)
                self.max_pool4 = nn.MaxPool2d(2,stride=2)
                self.fc1 = nn.Linear(2048,512)
                self.dropout = nn.Dropout(p=0.5)
                self.fc2 = nn.Linear(512,4)

            def forward(self,x):
                x = self.max_pool1(F.relu(self.bn3(self.conv2(F.relu(self.bn2(self.conv1(self.bn1(x))))))))
                x = self.max_pool2(F.relu(self.bn5(self.conv4(F.relu(self.bn4(self.conv3(x)))))))
                x = self.max_pool3(F.relu(self.bn8(self.conv7(F.relu(self.bn7(self.conv6(F.relu(self.bn6(self.conv5(x))))))))))
                x = self.max_pool4(F.relu(self.bn11(self.conv10(F.relu(self.bn10(self.conv9(F.relu(self.bn9(self.conv8(x))))))))))
                x = x.view(-1,2048)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
            
        def run_code_for_training_with_CrossEntropy(self, net):
            self.path_saved_model="praneet_noise.pth"
            filename_for_out1 = "./logs/noise_performance_numbers_" + str(self.epochs) + "_label_"+self.dataset_file[:-3]+".txt"
            FILE1 = open(filename_for_out1, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(),
                         lr=self.learning_rate, momentum=self.momentum)
            for epoch in range(self.epochs):
                running_loss_labeling = 0.0
                for i, data in enumerate(self.train_dataloader):
                    inputs, labels = data['image'],  data['label']
                    inputs = inputs.to(self.device)
                    labels = labels.view(8,1)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    labels=torch.squeeze(labels)
                    if labels.dim()==0:
                        loss_labeling=criterion(outputs,labels.unsqueeze(0))
                    else:
                        loss_labeling = criterion(outputs,labels)
                    loss_labeling.backward()
                    optimizer.step()
                    running_loss_labeling += loss_labeling.item()
                avg_loss_labeling = running_loss_labeling / float(500)
                print("\n[epoch:%d,  loss_labeling: %.3f  " % (epoch + 1,  avg_loss_labeling))
                FILE1.write("%.3f\n" % avg_loss_labeling)
                FILE1.flush()
                running_loss_labeling = 0.0    
                self.save_model(net)
                self.run_code_for_testing_classification()
            print("\nFinished Training\n")

        def save_model(self, model):
        
        #    Save the trained model to a disk file
            
            torch.save(model.state_dict(), self.path_saved_model)

        def run_code_for_testing_classification(self,net):
       #     net = NoiseClassifier.NoiseNet()
            self.path_saved_model="praneet_noise.pth"
            net.load_state_dict(torch.load(self.path_saved_model))
            correct = 0
            total = 0
            noise=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            with torch.no_grad():
                for i,data in enumerate(self.test_dataloader):
                    images,  labels = data['image'],  data['label']
                    labels = labels.tolist()
                    outputs = net(images)
                    outputs_label = outputs
                    _, predicted = torch.max(outputs_label.data, 1)
                    predicted = predicted.tolist()
                    for j in range(len(predicted)):
                            noise[int(labels[j])][int(predicted[j])]+=1
              #      print(labels, predicted)
                    total += len(labels)
                    correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
            print(noise)
            print("\n\n\nOverall accuracy of the network on the 4000 test images: %d %%" %
                                                                   (100 * correct / float(total)))





if __name__=="__main__":
   # assert args.train_tar, args.test_tar, "Please Specify train and test tar"
    detector = NoiseClassifier( )
    dataserver_train = NoiseClassifier.PurdueShapes5Dataset(
                                       train_or_test = 'train',
                                       dataroot = './',
    #                                   dataset_file = "PurdueShapes5-20-train.gz",
                                       dataset_files = args.train_tars
                                                                          )
    dataserver_test = NoiseClassifier.PurdueShapes5Dataset(
                                       train_or_test = 'test',
                                       dataroot='./',
    #                                   dataset_file = "PurdueShapes5-20-test.gz"
                                       dataset_files = args.test_tars
                                                                      )

  #  dataserver_train.load_and_augment_images()
    detector.dataserver_train = dataserver_train
   # dataserver_test.load_and_augment_images()
    detector.dataserver_test = dataserver_test
    detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)
    
    model = detector.NoiseNet()
     
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_layers = len(list(model.parameters())) 
    print("Number of trainable Parameters :"+str(model_params))
    print("Number of layers :"+str(model_layers))
    #dls.show_network_summary(model)
    detector.dataset_file = "noise"
    #detector.run_code_for_training_with_CrossEntropy(model)
    #detector.run_code_for_training_with_CrossEntropy_and_BCE_Losses(model)
    detector.run_code_for_testing_classification(model)
    
'''
