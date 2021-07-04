import os
import numpy as np
import pandas as pd
from typing import List
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


class DatasetFromImage(torch.utils.data.Dataset):
    '''Map-style dataset. Indices map to (img, label) E.g. img, y = data[0]
    
    User provides path of directory containing all images and
    annotation file in .csv format (with header), with first column
    containing image names and second column containing corresponding labels.
    
    Args:
        annot_file: (str) path to annotation file
        img_dir: (str) path to image directory
    '''
    def __init__(self, annot_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annot_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label


class Data:  
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        
    def load_image(self, annot_file, img_dir, transform=None, target_transform=None):
        # Instantiates DataSet object
        self.data = DatasetFromImage(annot_file, img_dir, transform, target_transform)
    
    def split(self, proportion=0.8):
        n = len(self.data)
        n_train = int(n * proportion)
        lengths = [n_train, n - n_train]
        self.train, self.test = torch.utils.data.random_split(self.data, lengths)
    
    def load(self, dir_path):
        self.train = torch.load(os.path.join(dir_path, "training.pt"))
        self.test = torch.load(os.path.join(dir_path, "test.pt"))
        
    def save(self, dir_path):
        # Saves DataSet object rather than tensor
        # Warning: Assumes that self.train and self.test are present
        train_path = os.path.join(dir_path, "training.pt")
        test_path = os.path.join(dir_path, "test.pt")
        
        if os.path.exists(dir_path):
            if os.path.exists(train_path) or os.path.exists(test_path):
                print("Over-writing files...")
                
            torch.save(self.train, train_path)
            torch.save(self.test, test_path)
        else:
            os.mkdir(dir_path)
            print(f'Directory created: {dir_path}')
            torch.save(self.train, train_path)
            torch.save(self.test, test_path)
        
#     def anonymise(self):
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)
    

# convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# multi-layer perceptron
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class Classifier:
    ERR_MSG = "Values not found within available options: "
    
    def __init__(self, model: str):
        if net == "mlp":
            self.net = MLP()
        elif net == "cnn":
            self.net = CNN()
        else:
            raise ValueError("Values not found within available options: 'mlp', 'cnn'")

    def training_step(self, batch):
        x, y = batch
        y_hat = self.net(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

#     def validation_step(self, batch, batch_idx):
#         loss, acc = self._shared_eval_step(batch, batch_idx)
#         metrics = {'val_acc': acc, 'val_loss': loss}
#         self.log_dict(metrics)
#         return metrics

#     def test_step(self, batch, batch_idx):
#         loss, acc = self._shared_eval_step(batch, batch_idx)
#         metrics = {'test_acc': acc, 'test_loss': loss}
#         self.log_dict(metrics)
#         return metrics

#     def _shared_eval_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.net(x)
#         loss = F.cross_entropy(y_hat, y)
#         acc = FM.accuracy(y_hat, y)
#         return loss, acc

    def predict_proba_step(self, batch):
        x, y = batch
        y_hat = self.net(x)
        return y_hat
    
    def predict_step(self, batch):
        x, y = batch
        y_hat = self.net(x)
        _, predicted_labels = torch.max(y_hat, 1)
        return predicted_labels

    def set_optimiser(self, optimiser="adam", lr=1e-3):
        if optimiser == "adam":
            self.optimiser = torch.optim.Adam(self.net.parameters(), lr=lr)
        elif optimiser == "sgd":
            self.optimiser = torch.optim.SGD(self.net.parameters(), lr=lr)
        else:
            raise ValueError(ERR_MSG + "'adam', 'sgd'")
    
    # Has to be set after optimiser is set
    def set_scheduler(self, scheduler="exponential", milestones=[3,8]):
        if scheduler == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=0.9)
        elif optimiser == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimiser, milestones=milestones, gamma=0.1)
        else:
            raise ValueError(ERR_MSG + "'exponential', 'multistep'")


            #torch.optim.lr_scheduler.MultiStepLR


class Trainer:
    def fit(self, model, train_dataloader, epochs=5):
        self.model = model
        # put self.model in train mode
        self.model.net.train()
        torch.set_grad_enabled(True) # test step: torch.freeze to freeze weights
        
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            for idx, batch in enumerate(train_dataloader):
                # zero the parameter gradients
                self.model.optimiser.zero_grad()
                # forward + backward + optimize
                loss = self.model.training_step(batch)
                loss.backward()
                self.model.optimiser.step()

                # print loss
                running_loss += loss.item()
                if idx % 1000 == 999:     # print every 1000 mini-batches
                    print(f"[{epoch + 1}, {idx + 1: 5d}] Loss = {running_loss / 1000:.3f}")
                    running_loss = 0.0
            
            if hasattr(self.model, "scheduler"):
                self.model.scheduler.step() #lr changes every epoch
                lr = self.model.scheduler.get_last_lr()[0]
                print(f"Learning rate = {lr:.5f}")
            
    def predict(self, dataloader) -> List[torch.Tensor]:
        # since we're not training, we don't need to calculate the gradients for our outputs
        self.model.net.train(False) # TODO: Check self.model.eval()
        
        with torch.no_grad():
            predictions = []
            for batch in dataloader:
                predictions.append(self.model.predict_step(batch))

        return predictions
    
    def count_classes(self, dataloader):
        predictions = self.predict(dataloader)
        all_pred = torch.cat(predictions)
        unique_classes = torch.unique(all_pred)
        shape = list(unique_classes.shape)
        
        return shape[0]
    
    def test(self, test_dataloader):
        # since we're not training, we don't need to calculate the gradients for our outputs
        self.model.net.train(False) # TODO: Check self.model.eval()
        
        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                # calculate outputs by running images through the network 
                outputs = self.model.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total
    
    def save_net(self, file):
        torch.save(self.model.net.state_dict(), file)
        
    def load_net(self, model ,file):
        self.model = model
        self.model.net.load_state_dict(torch.load(file))

