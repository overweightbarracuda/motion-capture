import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import math
import os

videos = []
for vid in os.listdir("C://JTT/"):
    loc = []
    with open("C://JTT/"+vid) as f:
        for line in f.readlines():
            loc.append([i.split(",") for i in line.split(";")])
    videos.append(loc)

trainRNN = []
labelRNN = []
for video in (videos):
    trainRNN.append(videos[videos.index(video)][:-1])
    labelRNN.append(videos[videos.index(video)][1:])
for video in trainRNN:
    for frame in video:
        for coords in frame:
            for coord in coords:
                trainRNN[trainRNN.index(video)][video.index(frame)][frame.index(coords)][coords.index(coord)] = int(trainRNN[trainRNN.index(video)][video.index(frame)][frame.index(coords)][coords.index(coord)])
for video in labelRNN:
    for frame in video:
        for coords in frame:
            for coord in coords:
                labelRNN[labelRNN.index(video)][video.index(frame)][frame.index(coords)][coords.index(coord)] = int(labelRNN[labelRNN.index(video)][video.index(frame)][frame.index(coords)][coords.index(coord)])

trainRNN = torch.tensor(trainRNN)                
labelRNN = torch.tensor(labelRNN)
                
                

def trainerRNN(e, lgr,b):    
    #criterion = nn.CrossEntropyLoss()
    #criterion = my_loss()
    criterion = nn.MSELoss()
    #optimizer = optim.SGD(net.parameters(), lr=lgr, momentum=0.9)
    optimizer = optim.Adam(RNNnet.parameters(), lr=lgr)
    for epoch in range(e):  # iterations over the training data
        running_loss = 0.0
        #inputs = trainRNN.reshape(4,len(trainRNN),47,58)
        #labels = labelRNN.reshape(4,len(labelRNN),47,58)
        inputs = trainRNN
        labels = labelRNN
        # zero the parameter gradients
        optimizer.zero_grad()
        # change the weights
        for i in range(int(len(inputs)/b)):
            outputs, hidden = RNNnet(inputs[i:i+b])
            outputs = torch.reshape(outputs,(b,12,47*58))
            loss = criterion(outputs, labels[i:i+b])
            loss.backward()
            optimizer.step()
                # print statistics
            running_loss += loss.item()
        print(epoch, running_loss)
    print('Finished Training')
    #print(torch.tensor(totouts).shape)
    return outputs
  class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden                 
RNNnet = RNN(input_size=33*3, output_size=33*3, hidden_dim=33*3, n_layers=1)
