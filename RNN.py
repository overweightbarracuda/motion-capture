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

video = []
with open("C://JTT/jab4") as f:
        for line in f.readlines():
            video.append([i.split(",") for i in line.split(";")])
            
trainRNN = video[:-1]
labelRNN = video[1:]
for frame in labelRNN:
    for coords in frame:
        for coord in coords:
            labelRNN[labelRNN.index(frame)][frame.index(coords)][coords.index(coord)] = int(labelRNN[labelRNN.index(frame)][frame.index(coords)][coords.index(coord)])
for frame in trainRNN:
    for coords in frame:
        for coord in coords:
            trainRNN[video.index(frame)][frame.index(coords)][coords.index(coord)] = int(trainRNN[video.index(frame)][frame.index(coords)][coords.index(coord)])
trainRNN = torch.tensor(trainRNN, dtype=torch.float32)
labelRNN = torch.tensor(labelRNN, dtype=torch.float32)

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
        self.fc2 = nn.Linear(hidden_dim, output_size)
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc2(self.fc(out))
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden


RNNnet = RNN(input_size=33*3, output_size=33*3, hidden_dim=33*3, n_layers=1)

def trainerRNN(e, lgr):
    criterion = nn.MSELoss()
    inputs, labels = trainRNN.reshape([42,33*3]), labelRNN.reshape([42,33*3])
    inputs = torch.unsqueeze(inputs, dim=0)
    labels = torch.unsqueeze(labels, dim=0)
    print(inputs.shape, labels.shape)
    optimizer = optim.Adam(RNNnet.parameters(), lr=lgr)
    for i in range(e+1):
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        output, hidden = RNNnet(inputs)
        loss = criterion(output, inputs)
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        if i%10 == 0:
            print('Epoch: {}/{}.............'.format(i, e), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
    return output, labels

out = trainerRNN(1000, 5)

for i in range(len(label[0])):
    print(torch.squeeze(label, dim=0)[0][i]-out[0][i])
