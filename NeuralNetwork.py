import torch
import pandas as pd
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
import cv2
import glob
import plotly.express as px
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Initialing compute device (use GPU if available).
print("Neural Network")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

brain_tumor = pd.read_csv(f'dataset/Brain Tumor.csv')
class_brain_tumor = brain_tumor.loc[:, ['Class']]
target = class_brain_tumor['Class'].values

image_data = []
for name in sorted(glob.glob('dataset/Brain Tumor/*'), key=len):
    im = cv2.imread(name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).flatten()
    #im = [im.mean(), im.std()]
    #im = torch.tensor(im)
    image_data.append(im)
    pass

image_data = torch.tensor(image_data)
(x_train, x_test, y_train, y_test) = train_test_split(image_data,target, test_size=0.25, random_state=42)

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)


trainset = TensorDataset(x_train, y_train)
testset = TensorDataset(x_test, y_test)
labeled_dataset = TensorDataset(x_labeled,y_labeled)
unlabeled_dataset = TensorDataset(x_unlabeled,y_unlabeled)

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=128, shuffle=False)
labeled_dataloader = DataLoader(labeled_dataset, batch_size=128, shuffle=True)

print(x_train.shape, x_test.shape)
pass