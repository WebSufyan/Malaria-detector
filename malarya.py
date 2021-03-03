import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

#proccessing images for the network
train_transformer = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transformer = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

root = 'C:\\Users\\User\\OneDrive\\Bureau\\coding\\ARTIFICIAL INTELLIGENCE PROJECTS\\Malarya detector in blood cell with pytorch'

train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transformer)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=train_transformer)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

class_names = train_data.classes

#determining the flattened shape of the input to flatten it so we can pass it to the linear layer
# for i, (image, label) in enumerate(train_data):
#     break
# conv1 = nn.Conv2d(3, 6, 3, 1)
# conv2 = nn.Conv2d(6, 16, 3, 1)
# x = image.view(1, 3, 32, 32)
# conv1 = F.relu(conv1(x))
# pool1 = F.max_pool2d(conv1, 2, 2)
# conv2 = F.relu(conv2(pool1))
# pool2 = F.max_pool2d(conv2, 2, 2)

#defining the model
class malarya_detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv1_drop = nn.Dropout2d(p=0.3)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.conv2_drop = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(6*6*16, 100)
        self.fc2 = nn.Linear(100, 80)
        self.fc3 = nn.Linear(80, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv1_drop(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2_drop(x)
        x = x.view(-1, 6*6*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

GPU = torch.device('cuda:0')
torch.manual_seed(101)

model = malarya_detector().to(GPU)
        
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
train_losses = []
test_losses = []
train_correct = []
test_correct = []

#training the model
for i in tqdm(range(epochs)):
    train_corr = 0
    test_corr = 0
    for b, (x_train, y_train) in enumerate(train_loader):
        y_pred = model(x_train.to(GPU))
        cost = loss(y_pred, y_train.to(GPU))
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        predicted = torch.argmax(y_pred, 1)
        batch_corr = (predicted == y_train.to(GPU)).sum()
        train_corr += batch_corr
        if b%200 == 1:
            print(f'epoch :{i}, batch :{b} [{10*b}],loss :{cost}, accuracy :{train_corr.item()*100/(10*b):7.3}%')
    train_losses.append(cost)
    train_correct.append(train_corr)
        
    with torch.no_grad():
        for b, (x_test, y_test) in enumerate(test_loader):
            y_eval = model(x_test.to(GPU))
            predicted = torch.argmax(y_eval, 1)
            test_batch = (predicted == y_test.to(GPU)).sum()
            test_corr += test_batch
            
    loss2 = loss(y_eval, y_test.to(GPU))
    test_losses.append(loss2)
    test_correct.append(test_corr)

# the model test accuracy is 91% 

#saving the model
# torch.save(model.state_dict(), 'malarya_detector_91%.pt')















