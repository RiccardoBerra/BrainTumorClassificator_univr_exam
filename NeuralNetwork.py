import torch
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
