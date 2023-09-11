import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

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
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)#.flatten()
    #im = [im.mean(), im.std()]
    #im = torch.tensor(im)
    image_data.append(im)
    pass

image_data = np.array(image_data)
image_data = image_data / 255.0

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[image_data.mean()], std=[image_data.std()])]
)

(x_train, x_test, y_train, y_test) = train_test_split(image_data,target, test_size=0.25, random_state=42)

# Apply the transformation to the training and testing sets
x_train_t = torch.stack([transform(x) for x in x_train])
x_test_t = torch.stack([transform(x) for x in x_test])

# Convert labels to PyTorch tensors
y_train_t = torch.tensor(y_train)
y_test_t = torch.tensor(y_test)

# Create PyTorch TensorDatasets for train and test
trainset = TensorDataset(x_train_t, y_train_t)
testset = TensorDataset(x_test_t, y_test_t)


train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
#unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=128, shuffle=False)
#labeled_dataloader = DataLoader(labeled_dataset, batch_size=128, shuffle=True)

print(x_train.shape, x_test.shape)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)

        # Pooling (Max).
        self.maxpool = nn.MaxPool2d(kernel_size=3)

        # Fully connected layers.
        self.fc1 = nn.Linear(189728, 128) #TODO: Check here
        #self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout.
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Convolution block 1.
        out = F.relu(self.conv1(x))

        # Convolution block 2 .
        out = F.relu(self.conv2(out))

        # Convolution block 3.
        out = F.relu(self.conv3(out))
        out = self.maxpool(out)
        out = self.dropout2d(out)

        # Create dense vector representation.
        # (BS, 32, 6, 6) - > (BS, 32 * 6 * 6).
        out = out.view(out.size(0), -1)

        # Linear (FC) layer.
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

model = CNN().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

losses = []
accuracies = []

num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Imposta il modello in modalità di training.
    predictions = []
    ground_truth = []
    total_loss = 0.0

    for images, labels in train_dataloader:  # Itera attraverso i batch dei dati di addestramento.
        images, labels = images.to(device), labels.to(device)  # Sposta i dati su GPU se disponibile.
        images = images.float()  # Converti l'input in tipo "float"

        optimizer.zero_grad()  # Azzera i gradienti accumulati dai passaggi precedenti.

        outputs = model(images)  # Esegui l'inoltro dei dati attraverso il modello.

        loss = loss_fn(outputs, labels)  # Calcola la loss.

        loss.backward()  # Calcola i gradienti.

        optimizer.step()  # Aggiorna i pesi del modello.

        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.detach().cpu().numpy().flatten().tolist())
        ground_truth.extend(labels.detach().cpu().numpy().flatten().tolist())
        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    losses.append(average_loss)
    if epoch % 20 == 0:
      accuracy = accuracy_score(ground_truth, predictions)
      accuracies.append(accuracy)
      print('Epoch: {}. Loss: {}. Accuracy (on trainset/self): {}'.format(epoch, loss.item(), accuracy))

model.eval()  # Imposta il modello in modalità di valutazione.

predictions = []
ground_truth = []

with torch.no_grad():  # Disabilita il calcolo dei gradienti durante la valutazione.
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        images = images.float()  # Converti l'input in tipo "float"
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        predictions.extend(predicted.cpu().numpy())
        ground_truth.extend(labels.cpu().numpy())



torch.save(model.state_dict(), 'model_weights_o.pth')
accuracy = accuracy_score(ground_truth, predictions)
print('Accuracy on test set:', accuracy)
print(confusion_matrix(ground_truth, predictions))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(len(losses)), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(0, num_epochs, 20), accuracies, label='Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()