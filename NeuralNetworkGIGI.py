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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

# Download the MNIST dataset using the PyTorch API.
train_dataset = MNIST(root='./data', train=True, transform=transforms.Normalize((0.1307,), (0.3081,)), download=True)
test_dataset = MNIST(root='./data', train=False, transform=transforms.Normalize((0.1307,), (0.3081,)), download=True)

# Load the trainset as a separate array and divide it into labeled and unlabeled partitions.
x_train, y_train = train_dataset.data, train_dataset.targets
x_test, y_test = test_dataset.data, test_dataset.targets

# Print out the dimensionality of the input images of both sets.
print(x_train.shape, x_test.shape)

# To start, we need to pose the MNIST digit recognition problem as a semi-supervised learning problem.
# To do this we'll set a number of samples to use in the first training procedure. Based on this number
# we'll take a number of labels per class and then train a classifier on this sub-sample.
num_train_samples = 1000
samples_per_class = int(num_train_samples/10)

# Based on the number of training samples, we can separate the data into two factions: unlabeled and labeled (both x and y).
# Hint: Keep in mind that all these operations are done on the training data!
# You should always treat your testing data as invisible and use it only for what the name suggests, i.e. testing.

# 1. Get number of unique classes.
unique_classes = y_train.unique()

# 2. Randomly choose samples_per_class elements for each unique class (for bonus XP try to do this in one line).
# Hint: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
# Hint: https://pytorch.org/docs/stable/generated/torch.where.html
# You should end up with an array of num_train_samples indices that reference the labeled training samples.
subsample_idx = [np.random.choice(torch.where(y_train == mnist_class)[0], size=samples_per_class, replace=False) for mnist_class in unique_classes]
subsample_idx = np.array(subsample_idx).flatten()

# 3. Separate the training datasets into two subsets: labeled and unlabeled.
# Hint: create a boolean mask to easily separate the subsets.
unlabeled_mask = np.ones(y_train.shape[0], dtype=bool)
unlabeled_mask[subsample_idx] = False
x_labeled, x_unlabeled = x_train[subsample_idx, :], x_train[unlabeled_mask, :]
y_labeled, y_unlabeled = y_train[subsample_idx], y_train[unlabeled_mask]

# 4. Print out the shapes of the subsets (Keep in mind now we are working with tensors).
print(x_labeled.shape, y_labeled.shape, x_unlabeled.shape, y_unlabeled.shape)

# Create PyTorch Datasets and dataloader objects using the subsets.
# Hint: https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset.
# Hint: https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler.
trainset = TensorDataset(x_train, y_train)
testset = TensorDataset(x_test, y_test)
labeled_dataset = TensorDataset(x_labeled,y_labeled)
unlabeled_dataset = TensorDataset(x_unlabeled,y_unlabeled)

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=128, shuffle=False)
labeled_dataloader = DataLoader(labeled_dataset, batch_size=128, shuffle=True)

# Create a basic CNN classifier using PyTorch that can fit the MNIST data.
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
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
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

# Train CNN model on labeled dataset.
num_epochs = 100
for i in range(num_epochs):
    model.train()
    predictions = []
    ground_truth = []
    for images, labels in labeled_dataloader:
        # Add dimension for Nr. channels (we don't have it here because of Grayscale) and transform the images tensor into a float.
        images, labels = images.unsqueeze(1).float().to(device), labels.to(device)

        # Clear gradients w.r.t. parameters.
        optimizer.zero_grad()

        # Forward pass to get output.
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss.
        loss = loss_fn(outputs, labels)

        # Getting gradients w.r.t. parameters.
        loss.backward()

        # Updating parameters.
        optimizer.step()

        # Get predictions from the maximum value and append them to calculate accuracy later.
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.detach().cpu().numpy().flatten().tolist())
        ground_truth.extend(labels.detach().cpu().numpy().flatten().tolist())

    if i % 20 == 0:
      accuracy = accuracy_score(ground_truth, predictions)
      print('Epoch: {}. Loss: {}. Accuracy (on trainset/self): {}'.format(i, loss.item(), accuracy))


# Check where our CNN has made mistakes by using a confusion matrix.
print(confusion_matrix(ground_truth, predictions))

# Predict the pseudo-labels of the unlabeled dataset.
pseudo_labels = []
ground_truth = []
for images, labels in unlabeled_dataloader:
  model.eval()
  with torch.no_grad():
        # Add dimension for Nr. channels (we don't have it here because of Grayscale) and transform the images tensor into a float.
        images, labels = images.unsqueeze(1).float().to(device), labels.to(device)

        # Forward pass to get output.
        outputs = model(images)

        # Get predictions from the maximum value and append them to calculate accuracy later.
        _, predicted = torch.max(outputs.data, 1)
        pseudo_labels.extend(predicted.detach().cpu().numpy().flatten().tolist())
        ground_truth.extend(labels.detach().cpu().numpy().flatten().tolist())

# Calculate the accuracy of the pseudo-labeling procedure (we can compute this since we actually have a fully labeled dataset).
accuracy = accuracy_score(ground_truth, pseudo_labels)
print('Accuracy of pseudo-labels: {}'.format(accuracy))


# Check where our CNN has mistaken the pseudo-labels by using a confusion matrix.
print(confusion_matrix(y_unlabeled, pseudo_labels))

# Create a new training set using the new pseudo labels and the old supervised labels.
# Print out the shapes (you should get the same ones of the original MNIST dataset).
# HINT: https://pytorch.org/docs/stable/generated/torch.vstack.html.
new_x = torch.vstack((x_labeled, x_unlabeled))
new_y = torch.vstack((y_labeled.view(-1, 1), torch.LongTensor(pseudo_labels).view(-1, 1))).squeeze()
print(new_x.shape, new_y.shape)

# Build a new dataset using the above tensors
pseudo_dataset = TensorDataset(new_x, new_y)
pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=200, shuffle=True)

# Retrain the model on the pseudo-labeled dataset and test out the performance on the testing set (real labels).
new_model = CNN().to(device)
optimizer = torch.optim.SGD(new_model.parameters(), lr=1e-3, momentum=0.9)
num_epochs = 100
for i in range(num_epochs):
    new_model.train()
    for images, labels in pseudo_dataloader:
        # Add dimension for Nr. channels (we don't have it here because of Grayscale) and transform the images tensor into a float.
        images, labels = images.unsqueeze(1).float().to(device), labels.to(device)

        # Clear gradients w.r.t. parameters.
        optimizer.zero_grad()

        # Forward pass to get output.
        outputs = new_model(images)

        # Calculate Loss: softmax --> cross entropy loss.
        loss = loss_fn(outputs, labels)

        # Getting gradients w.r.t. parameters.
        loss.backward()

        # Updating parameters.
        optimizer.step()

    # Evaluate directly on testset.
    if i % 10 == 0:
        predictions = []
        ground_truth = []
        for images, labels in test_dataloader:
            new_model.eval()
            with torch.no_grad():
                # Add dimension for Nr. channels (we don't have it here because of Grayscale) and transform the images tensor into a float.
                images, labels = images.unsqueeze(1).float().to(device), labels.to(device)

                # Forward pass to get output.
                outputs = new_model(images)

                # Get predictions from the maximum value and append them to calculate accuracy later.
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.detach().cpu().numpy().flatten().tolist())
                ground_truth.extend(labels.detach().cpu().numpy().flatten().tolist())

        accuracy = accuracy_score(ground_truth, predictions)
        print('Epoch: {}. Loss: {}. Accuracy (on testing set): {}'.format(i, loss.item(), accuracy))