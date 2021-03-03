import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


image_transformer = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

root = 'C:\\Users\\User\\OneDrive\\Bureau\\coding\\ARTIFICIAL INTELLIGENCE PROJECTS\\Malarya detector in blood cell with pytorch'

single_image = datasets.ImageFolder(os.path.join(root, 'malarya_image'), transform=image_transformer)
image_loader = DataLoader(single_image)

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


# reshaping image and proccessing it for prediction
def get_image(img):
    img = cv2.imread(img,0)
    pil_img = Image.fromarray(img).convert('RGB')
    transformed = image_transformer(pil_img)
    resize = transformed.view(1, 3, 32, 32)
    return resize

model = malarya_detector()
model.load_state_dict(torch.load('malarya_detector_91%.pt'))

image = get_image('malarya_image\\sample\\Sickle-Cell-birth-disorders-article.jpg')
with torch.no_grad():
    z = model(image)
    predicted = torch.argmax(z, 1)














