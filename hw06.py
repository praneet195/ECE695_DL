''' 
    ***Please ensure you provide the correct sys.argv[1] argument while running each task i.e task1 for hw06_task1, task2 for hw06_task2 and task3 for hw06_task3***
    
################################################################################################################################################################################################################################################################
    HW06 Task 1
    
    Run: python2 hw06.py task1
   
    This code is based on Prof Kak's DLStudio and Gabriel Lloye's Git https://github.com/gabrielloye/GRU_Prediction. I've just used MSELoss instead of NLLLoss and used the ADAM optimizer instead of SGD. The originalk model trained on 43k one hot vectors gets a testing accuracy of about 85% on this setup i.e [(true pos + true neg)/2] 
    
    Please ensure the dataset is placed in a folder "data" in the current directory.

    In task1, we need to find alternative approaches to make a more efficient representations of the reviews.
    Here, 3 approches have been tried.
    
        1) Word2Idx : Each word in a review is being replaced by it's index in the vocabulary. Completely eliminates massive 43k one hot vectors but the performance of the model is horrible in this case. Results have not been reported in output.txt for this case. This method gets only a 50% accuracy.
    
        2) ShortVec: Here, I've removed all the unnecessary words from the vocabulary. By unnecessary, I mean filler words like 'a','an',the' and words that occur only once in the entire dataset. We reduced the length of the one hot vectors to about 10k from about 43k which is a significant reduction. I've posted the results of this approach in the output.txt file. We see a small drop in accuracy when compared to the model trained on the 43k one hot vectors, and also pre-trained Google Word2Vec model isn't required to run this. Accuracy is 83%. Results reported in output.txt.

        3) Google's Pre-trained Word2Vec: Here, we use google's pretrained word2vec model to generate vectors of size 300 for each word in the review. However, to run this we need the Google bin file found at https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit. I then used the vocab provided as the corpus. Results are excellent with an accuracy of 88%, but it requires gensim and a 1.5GB pre-trained model and hence I have not reported it's results in the output.txt. Please ask me for the file in order to run this Word2Vec model.
    
    In order to run one of the methods mentioned above, please set the third argument in the dataset_train and dataset_test lines of this code to either "shortvec", "wordidx", or "word2vec". If left empty, it'll run the original 43k one hot vector code as seen in the two references mentioned above.
    
    Futhermore, by removing the if conditions, we can also combine a few of the approaches to get better resultssuch as remove the unnecessary words and then lt google's word2vec decide the vector for the remaining words.
    
    Also, from here on out, I will use the ShortVec approach. This is because it makes the reviews manageable with slight decrease in accuracy works. Also removes the need to have gensim and to downlaod the 1.5GB Google pre-trained word2vec bin
    Please let me know if any pre-trained weights are required to reproduce the results in the output.txt

################################################################################################################################################################################################################################################################

    HW06 Task 2
    
    Run: python3 hw06.py task2
    
    In this task we had to improve the performance of the TEXTnetOrder2 using some new gating function and logic.
    These are the things that I tried:
    1) Changed the activation to sigmoid that showed a slight improvement in results.
    2) Accesed two previous layer hidden values that also slightly improved results.
    
    The original TEXTnetOrder2 gave me an accuracy of 62%.
    This modification gave me an accuracy of 67%.
 ################################################################################################################################################################################################################################################################
 
    HW06 Task 3
   
    Run: python3 hw06.py task3

    In this code, I've used the GRUNet and the shortvec representation from Task 1 to show how padding, truncating and averaging can help improve in sentiment analysis when batching is involved

    Please ensure the dataset is placed in a folder "data" in the current directory.

    There are three methods that can be used here:
    
        1)Padding: Use pad_collate in collate_fn for train dataloader. This pads each review in the batch to be of equal size i.e the size of the longest review
    
        2)Truncating: Use truncate_collate in collate_fn for train dataloader. This truncates each review in the batch to an equal size i.e length of the shortest review
    
        3) Averaging: Use avg_collate in collate_fn for train dataloader. This makes each review of equal size where the size is equal to the average of the length of the reviews in the batch.
        2) and 3) use features of 1) and hence they are interdependent.

    The results reported in output.txt are for Truncating which performed the best in my case i.e batch size 4, GRUNet and shortvec representations.
    I also ran the same procedures for the original 43k one hot vectore representation and saw that truncating performed the best there as well.
    I got accuracies of about 78% when I used the 43k vectors with truncating but saw an accuracy of 80% when I used short vector representation with truncating.
    Also, I have anotehr implementation where I use a hidden of size 4. The current case is more of a subdivision but it still implements the end goal of batch size becuase each review has its own hidden of size 1 and it's correct too. I've reported the case I got the best results for.
    Please ask me if weight files are required to reproduce the results obtained.
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
from PIL import ImageFilter
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import pymsgbox
import time

class SentimentAnalysisDataset(torch.utils.data.Dataset):
        def __init__(self, root , train_or_test,type1, dataset_file,word_dict={}):
                    super(SentimentAnalysisDataset, self).__init__()
                    self.train_or_test = train_or_test
                    root_dir = root
                    self.type = type1
                    self.train_or_test = train_or_test
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    self.word_dict= word_dict
                    if self.train_or_test is 'train':
                        if sys.version_info[0] == 3:
                            self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
                        self.indexed_dataset_train = []
                        for category in self.positive_reviews_train:
                            for review in self.positive_reviews_train[category]:
                                for k in review:
                                    if k not in self.word_dict.keys():
                                        self.word_dict[k]=0
                                    else:
                                        self.word_dict[k]+=1
                                self.indexed_dataset_train.append([review,  1])
                        for category in self.negative_reviews_train:
                            for review in self.negative_reviews_train[category]:
                                for k in review:
                                    if k not in self.word_dict.keys():
                                        self.word_dict[k]=0
                                    else:
                                        self.word_dict[k]+=1
                                self.indexed_dataset_train.append([review,  0])
                        random.shuffle(self.indexed_dataset_train)
                    elif self.train_or_test is 'test':
                        if sys.version_info[0] == 3:
                            self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
                        self.vocab = sorted(self.vocab)
                        self.indexed_dataset_test = []
                        for category in self.positive_reviews_test:
                            for review in self.positive_reviews_test[category]:
                                self.indexed_dataset_test.append([review, 1])
                        for category in self.negative_reviews_test:
                            for review in self.negative_reviews_test[category]:
                                self.indexed_dataset_test.append([review,  0])
                        random.shuffle(self.indexed_dataset_test)
                    if self.type == "shortvec":
                        self.update_indexed_datasets()
        
        def get_word_dict(self):
            return self.word_dict
        
        def update_indexed_datasets(self):
                self.vocab = []
                for i in self.word_dict.keys():
                    if self.word_dict[i] >2 and self.word_dict[i]<=9000:
                            self.vocab.append(i)
                self.vocab = sorted(self.vocab)
                if self.train_or_test == "train":
                    for i in range(len(self.indexed_dataset_train)):
                        old_review = self.indexed_dataset_train[i][0]
                        new_review = []
                        for k in old_review:
                            if k in self.vocab: 
                                new_review.append(k)
                        self.indexed_dataset_train[i][0]=new_review
                    random.shuffle(self.indexed_dataset_train)
                elif self.train_or_test == "test":
                    for i in range(len(self.indexed_dataset_test)):
                        old_review = self.indexed_dataset_test[i][0]
                        new_review = []
                        for k in old_review:
                            if k in self.vocab: 
                                new_review.append(k)
                        self.indexed_dataset_test[i][0]=new_review
                    random.shuffle(self.indexed_dataset_test)
                    
    
    
        def get_vocab_size(self):
                    return len(self.vocab)
    
        def one_hotvec_for_word(self, word):
                    word_index =  self.vocab.index(word)
                    hotvec = torch.zeros(1, len(self.vocab))
                    hotvec[0, word_index] = 1
                    return hotvec
    
        def review_to_tensor(self, review):
                    review_tensor = torch.zeros(len(review), len(self.vocab))
                    for i,word in enumerate(review):
                        if self.type == "wordidx":
                            review_tensor[i,:] = self.vocab.index(word)
                        else:
                            review_tensor[i,:] = self.one_hotvec_for_word(word)
                    return review_tensor
    
        def sentiment_to_tensor(self, sentiment):
                    """
                    Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
                    sentiment and 1 for positive sentiment.  We need to pack this value in a
                    two-element tensor.
                    """
                    sentiment_tensor = torch.zeros(2)
                    if sentiment is 1:
                        sentiment_tensor[1] = 1
                    elif sentiment is 0:
                        sentiment_tensor[0] = 1
                    sentiment_tensor = sentiment_tensor.type(torch.long)
                    return sentiment_tensor
    
        def __len__(self):
                    if self.train_or_test is 'train':
                        return len(self.indexed_dataset_train)
                    elif self.train_or_test is 'test':
                        return len(self.indexed_dataset_test)
    
        def __getitem__(self, idx):
                    sample = self.indexed_dataset_train[idx] if self.train_or_test is 'train' else self.indexed_dataset_test[idx]
                    review = sample[0]
                    review_sentiment = sample[1]
                    review_sentiment = self.sentiment_to_tensor(review_sentiment)
                    review_tensor = self.review_to_tensor(review)
                    sample = {'review'       : review_tensor,
                              'sentiment'    : review_sentiment }
                    return sample
    

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
        
class TEXTnetOrder2(nn.Module):
                def __init__(self, input_size, hidden_size, output_size,device):
                    super(TEXTnetOrder2, self).__init__()
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.output_size = output_size
                    self.combined_to_hidden = nn.Linear(input_size + 2*hidden_size, hidden_size)
                    self.combined_to_hidden2 = nn.Linear(input_size + 3*hidden_size, hidden_size)
                    self.combined_to_middle = nn.Linear(input_size + 2*hidden_size, 100)
                    self.combined_to_middle2 = nn.Linear(input_size + 3*hidden_size, 100)
                    self.middle_to_out = nn.Linear(100, output_size)
                    self.logsoftmax = nn.LogSoftmax(dim=1)
                    self.dropout = nn.Dropout(p=0.5)
                    self.bias = nn.Parameter(torch.empty(hidden_size).normal_(mean=0,std=1))
                    self.prev_hidden = torch.zeros(hidden_size)
                    
                    # for the cell
                    self.cell = torch.zeros(1, hidden_size).to(device)
                    self.linear_for_cell = nn.Linear(hidden_size, hidden_size)
                    
                    self.linear_cell_i = nn.Linear(input_size,hidden_size)
                    self.linear_cell_h = nn.Linear(hidden_size,hidden_size)
    
                def forward(self, input, hidden,k):
                    if k==0:
                        input_clone = input.clone()
                        combined = torch.cat((input, hidden, self.cell), 1)
                        hidden = self.combined_to_hidden(combined)
                        out = self.combined_to_middle(combined)
                        out = torch.nn.functional.relu(out)
                        out = self.dropout(out)
                        out = self.middle_to_out(out)
                        hidden_clone = hidden.clone()
                        lc = self.linear_cell_i(input_clone) +self.linear_cell_h(hidden_clone)  +self.bias
                        self.cell = torch.tanh(self.linear_for_cell(lc)).detach()
                        self.prev_hidden = self.linear_cell_h(hidden_clone).detach()
                    else:
                        input_clone = input.clone()
                        combined = torch.cat((input, hidden, self.prev_hidden, self.cell), 1)
                        hidden = self.combined_to_hidden2(combined)
                        out = self.combined_to_middle2(combined)
                        out = torch.nn.functional.relu(out)
                        out = self.dropout(out)
                        out = self.middle_to_out(out)
                        hidden_clone = hidden.clone()
                        lc = self.linear_cell_i(input_clone)+ self.linear_cell_h(hidden_clone)  +self.bias
                        self.cell = torch.tanh(self.linear_for_cell(lc)).detach()
                        self.prev_hidden = self.linear_cell_h(hidden_clone).detach()

                    return out,hidden

    
def training(net,device,train_dataloader,epochs):
                filename_for_out = "performance_numbers_" + str(epochs) + ".txt"
                FILE = open(filename_for_out, 'w')
                net = copy.deepcopy(net)
                net = net.to(device)
                criterion = nn.MSELoss()
                accum_times = []
                optimizer = optim.Adam(net.parameters(),
                             lr=1e-3)
                net.train()
                for epoch in range(epochs):
                    print("")
                    running_loss = 0.0
                    start_time = time.clock()
                    for i, data in enumerate(train_dataloader):
                        review_tensor,sentiment = data['review'],  data['sentiment']
                        review_tensor = review_tensor.to(device)
                        sentiment = sentiment.to(device)
                        sentiment = sentiment.float()
                        optimizer.zero_grad()
                        hidden = net.init_hidden(1).to(device)
                        for k in range(review_tensor.shape[1]):
                            output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                        ## If using NLLLoss, CrossEntropyLoss
                    #    loss = criterion(output, torch.argmax(sentiment, 1))
                        loss = criterion(output, sentiment)
                        running_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        if i % 100 == 99:
                            avg_loss = running_loss / float(100)
                            current_time = time.clock()
                            time_elapsed = current_time-start_time
                            print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.3f" % (epoch+1,i+1, time_elapsed,avg_loss))
                            accum_times.append(current_time-start_time)
                            FILE.write("%.3f\n" % avg_loss)
                            FILE.flush()
                            running_loss = 0.0
                print("Total Training Time: {}".format(str(sum(accum_times))))
                print("\nFinished Training\n")
                torch.save(net,"hw06.pt")
    
    
    
def testing( net, hidden_size,test_dataloader,device):
                #net.load_state_dict(torch.load(model_path))
                net = torch.load("hw06.pt")
                net.to(device)
                classification_accuracy = 0.0
                negative_total = 0
                positive_total = 0
                confusion_matrix = torch.zeros(2,2)
                with torch.no_grad():
                    for i, data in enumerate(test_dataloader):
                        review_tensor,sentiment = data['review'], data['sentiment']
                        review_tensor = review_tensor.to(device)
                        sentiment = sentiment.to(device)
                        sentiment = sentiment.float()
                        hidden = net.init_hidden(1)
                        for k in range(review_tensor.shape[1]):
                            output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                        predicted_idx = torch.argmax(output).item()
                        gt_idx = torch.argmax(sentiment).item()
                        if i % 100 == 99:
                            print("   [i=%d]    predicted_label=%d       gt_label=%d\n\n" % (i+1, predicted_idx,gt_idx))
                        if predicted_idx == gt_idx:
                            classification_accuracy += 1
                        if gt_idx is 0:
                            negative_total += 1
                        elif gt_idx is 1:
                            positive_total += 1
                        confusion_matrix[gt_idx,predicted_idx] += 1
                out_percent = np.zeros((2,2), dtype='float')
                print("\n\nNumber of positive reviews tested: %d" % positive_total)
                print("\n\nNumber of negative reviews tested: %d" % negative_total)
                print("\n\nDisplaying the confusion matrix:\n")
                out_str = "                      "
                out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
                print(out_str + "\n")
                for i,label in enumerate(['true negative', 'true positive']):
                    out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
                    out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
                    out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
                    out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
                    out_str = "%12s:  " % label
                    for j in range(2):
                        out_str +=  "%18s" % out_percent[i,j]
                    print(out_str)
                    
    
def training_textnet(net,device,train_dataloader,epochs,hidden_size):
                filename_for_out = "performance_numbers_" + str(epochs) + ".txt"
                FILE = open(filename_for_out, 'w')
                net = copy.deepcopy(net)
                net = net.to(device)
                criterion = nn.MSELoss()
                accum_times = []
                optimizer = optim.Adam(net.parameters(),
                             lr=1e-4)
                net.train()
                for epoch in range(epochs):
                    print("")
                    running_loss = 0.0
                    start_time = time.clock()
                    for i, data in enumerate(train_dataloader):
                        review_tensor,sentiment = data['review'],  data['sentiment']
                        review_tensor = review_tensor.to(device)
                        sentiment = sentiment.to(device)
                        sentiment = sentiment.float()
                        optimizer.zero_grad()
                        hidden = torch.zeros(1,hidden_size)
                        hidden = hidden.to(device)
                        inputt = torch.zeros(1,review_tensor.shape[2])
                        inputt = inputt.to(device)
                        for k in range(review_tensor.shape[1]):
                                inputt[0,:] = review_tensor[0,k]
                                output, hidden = net(inputt, hidden,k)
    
                        ## If using NLLLoss, CrossEntropyLoss
                    #    loss = criterion(output, torch.argmax(sentiment, 1))
                        loss = criterion(output, sentiment)
                        running_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        if i % 100 == 99:
                            avg_loss = running_loss / float(100)
                            current_time = time.clock()
                            time_elapsed = current_time-start_time
                            print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.3f" % (epoch+1,i+1, time_elapsed,avg_loss))
                            accum_times.append(current_time-start_time)
                            FILE.write("%.3f\n" % avg_loss)
                            FILE.flush()
                            running_loss = 0.0
                print("Total Training Time: {}".format(str(sum(accum_times))))
                print("\nFinished Training\n")
                torch.save(net,"hw06.pt")
    
    
    
def testing_textnet( net, hidden_size,test_dataloader,device):
                #net.load_state_dict(torch.load(model_path))
                net = torch.load("hw06.pt")
                net.to(device)
                classification_accuracy = 0.0
                negative_total = 0
                positive_total = 0
                confusion_matrix = torch.zeros(2,2)
                with torch.no_grad():
                    for i, data in enumerate(test_dataloader):
                        review_tensor,sentiment = data['review'], data['sentiment']
                        review_tensor = review_tensor.to(device)
                        sentiment = sentiment.to(device)
                        sentiment = sentiment.float()
                        hidden = torch.zeros(1,hidden_size)
                        hidden = hidden.to(device)
                        inputt = torch.zeros(1,review_tensor.shape[2])
                        inputt = inputt.to(device)
                        for k in range(review_tensor.shape[1]):
                            inputt[0,:] = review_tensor[0,k]
                            output, hidden = net(inputt, hidden,k)
                        predicted_idx = torch.argmax(output).item()
                        gt_idx = torch.argmax(sentiment).item()
                        if i % 100 == 99:
                            print("   [i=%d]    predicted_label=%d       gt_label=%d\n\n" % (i+1, predicted_idx,gt_idx))
                        if predicted_idx == gt_idx:
                            classification_accuracy += 1
                        if gt_idx is 0:
                            negative_total += 1
                        elif gt_idx is 1:
                            positive_total += 1
                        confusion_matrix[gt_idx,predicted_idx] += 1
                out_percent = np.zeros((2,2), dtype='float')
                print("\n\nNumber of positive reviews tested: %d" % positive_total)
                print("\n\nNumber of negative reviews tested: %d" % negative_total)
                print("\n\nDisplaying the confusion matrix:\n")
                out_str = "                      "
                out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
                print(out_str + "\n")
                for i,label in enumerate(['true negative', 'true positive']):
                    out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
                    out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
                    out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
                    out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
                    out_str = "%12s:  " % label
                    for j in range(2):
                        out_str +=  "%18s" % out_percent[i,j]
                    print(out_str)
                    
def training_batch(net,device,train_dataloader,epochs):
                filename_for_out = "performance_numbers_" + str(epochs) + ".txt"
                FILE = open(filename_for_out, 'w')
                net = copy.deepcopy(net)
                net = net.to(device)
                criterion = nn.MSELoss()
                accum_times = []
                optimizer = optim.Adam(net.parameters(),
                             lr=1e-3)
                net.train()
                for epoch in range(epochs):
                    print("")
                    running_loss = 0.0
                    start_time = time.clock()
                    for i, data in enumerate(train_dataloader):
                        review_tensor =[]
                        sentiment=[]
                        for mm in data:
                            review_tensor.append(torch.unsqueeze(mm['review'],0))
                            sentiment.append(mm['sentiment'])
                        optimizer.zero_grad()
                        out_stack=[]
                        for mm in range(len(review_tensor)):
                            hidden = net.init_hidden(1).to(device)
                            rev = review_tensor[mm].to(device)
                            for k in range(rev.shape[1]):
                                output, hidden = net(torch.unsqueeze(torch.unsqueeze(rev[0,k],0),0), hidden)
                            out_stack.append(output)
                        ## If using NLLLoss, CrossEntropyLoss
                    #    loss = criterion(output, torch.argmax(sentiment, 1))
                        out_stack = torch.squeeze(torch.stack(out_stack)).to(device)
                        sentiment = torch.stack(sentiment).to(device)
                        sentiment = sentiment.float()
                        loss = criterion(out_stack, sentiment)
                        running_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        if i % 100 == 99:
                            avg_loss = running_loss / float(100)
                            current_time = time.clock()
                            time_elapsed = current_time-start_time
                            print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.3f" % (epoch+1,i+1, time_elapsed,avg_loss))
                            accum_times.append(current_time-start_time)
                            FILE.write("%.3f\n" % avg_loss)
                            FILE.flush()
                            running_loss = 0.0
                print("Total Training Time: {}".format(str(sum(accum_times))))
                print("\nFinished Training\n")
                torch.save(net,"hw06.pt")
    
    
    
def testing_batch( net, hidden_size,test_dataloader,device):
                #net.load_state_dict(torch.load(model_path))
                net = torch.load("hw06.pt")
                net.to(device)
                classification_accuracy = 0.0
                negative_total = 0
                positive_total = 0
                confusion_matrix = torch.zeros(2,2)
                with torch.no_grad():
                    for i, data in enumerate(test_dataloader):
                        review_tensor,sentiment = data['review'], data['sentiment']
                        review_tensor = review_tensor.to(device)
                        sentiment = sentiment.to(device)
                        sentiment = sentiment.float()
                        hidden = net.init_hidden(1)
                        for k in range(review_tensor.shape[1]):
                            output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                        predicted_idx = torch.argmax(output).item()
                        gt_idx = torch.argmax(sentiment).item()
                        if i % 100 == 99:
                            print("   [i=%d]    predicted_label=%d       gt_label=%d\n\n" % (i+1, predicted_idx,gt_idx))
                        if predicted_idx == gt_idx:
                            classification_accuracy += 1
                        if gt_idx is 0:
                            negative_total += 1
                        elif gt_idx is 1:
                            positive_total += 1
                        confusion_matrix[gt_idx,predicted_idx] += 1
                out_percent = np.zeros((2,2), dtype='float')
                print("\n\nNumber of positive reviews tested: %d" % positive_total)
                print("\n\nNumber of negative reviews tested: %d" % negative_total)
                print("\n\nDisplaying the confusion matrix:\n")
                out_str = "                      "
                out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
                print(out_str + "\n")
                for i,label in enumerate(['true negative', 'true positive']):
                    out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
                    out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
                    out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
                    out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
                    out_str = "%12s:  " % label
                    for j in range(2):
                        out_str +=  "%18s" % out_percent[i,j]
                    print(out_str)
    
                    
def pad_collate(batch):
        rev = []
        for x in batch:
            rev.append(x['review'])
        rev_pad = torch.nn.utils.rnn.pad_sequence(rev, batch_first=True, padding_value=0)
        for i,x in enumerate(batch):
            x['review'] = rev_pad[i]
        return batch
    
def avg_collate(batch):
        rev = []
        sum_len = 0
        for x in batch:
            sum_len+=len(x['review'])
            rev.append(x['review'])
        avg_len = int(sum_len/len(batch))
        for i,x in enumerate(batch):
                x['review'] = x['review'][:avg_len]
        pad_collate(batch)
        return batch
    
def truncate_collate(batch):
        rev = []
        min_len = 9999
        for x in batch:
            if len(x['review']) < min_len:
                min_len = len(x['review'])
            rev.append(x['review'])
        for i,x in enumerate(batch):
                x['review'] = x['review'][:min_len]
        return batch
    

def run_hw06_task1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_train = SentimentAnalysisDataset("./data/","train","shortvec","sentiment_dataset_train_200.tar.gz")
    train_word_dict = dataset_train.get_word_dict()
    dataset_test = SentimentAnalysisDataset("./data/","test","shortvec","sentiment_dataset_test_200.tar.gz",train_word_dict)
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                        batch_size=1,shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(dataset_test,
                               batch_size=1,shuffle=False, num_workers=4)    
    vocab_size = dataset_train.get_vocab_size()
    hidden_size = 256
    output_size = 2                            # for positive and negative sentiments
    n_layers = 2
    model = GRUNet(vocab_size, hidden_size, output_size, n_layers)
    number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))
    print("\n\nThe number of layers in the model: %d" % num_layers)
    print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
    print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)
    training(model,device,train_dataloader,2)
    testing(model,hidden_size,test_dataloader,device)

def run_hw06_task2():
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset_train = SentimentAnalysisDataset("./data/","train","shortvec","sentiment_dataset_train_200.tar.gz")
    train_dict  = dataset_train.get_word_dict()
    dataset_test = SentimentAnalysisDataset("./data/","test","shortvec","sentiment_dataset_test_200.tar.gz",train_dict)
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                        batch_size=1,shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(dataset_test,
                               batch_size=1,shuffle=False, num_workers=4)
    vocab_size = dataset_train.get_vocab_size()
    hidden_size = 256
    output_size = 2                            # for positive and negative sentiments
    model = TEXTnetOrder2(vocab_size, hidden_size, output_size,device)
    number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))
    print("\n\nThe number of layers in the model: %d" % num_layers)
    print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
    print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)
    training_textnet(model,device,train_dataloader,2,hidden_size)
    testing_textnet(model,hidden_size,test_dataloader,device)
    
def run_hw06_task3():    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_train = SentimentAnalysisDataset("./data/","train","shortvec","sentiment_dataset_train_200.tar.gz")
    train_word_dict = dataset_train.get_word_dict()
    dataset_test = SentimentAnalysisDataset("./data/","test","shortvec","sentiment_dataset_test_200.tar.gz",train_word_dict)
    batch_size = 4
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                        batch_size=batch_size,shuffle=True, num_workers=4,collate_fn = truncate_collate)
    test_dataloader = torch.utils.data.DataLoader(dataset_test,
                               batch_size=1,shuffle=False, num_workers=4)
    vocab_size = dataset_train.get_vocab_size()
    hidden_size = 256
    output_size = 2                            # for positive and negative sentiments
    n_layers = 2
    model = GRUNet(vocab_size, hidden_size, output_size, n_layers)
    number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))
    print("\n\nThe number of layers in the model: %d" % num_layers)
    print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
    print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)
    training_batch(model,device,train_dataloader,2)
    testing_batch(model,hidden_size,test_dataloader,device)
    
if __name__=="__main__":
    assert sys.argv[1], "Please specify sys.argv[1] as task1, task2 or task3"
    if sys.argv[1] == "task1":
        run_hw06_task1()
    if sys.argv[1] == "task2":
        run_hw06_task2()
    if sys.argv[1] == "task3":
        run_hw06_task3()

        


