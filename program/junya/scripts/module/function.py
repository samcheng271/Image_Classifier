import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision.datasets import VisionDataset
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
import re
import pandas as pd


class CustomTrainImageDataset(Dataset):
    # Initialization method to set up the dataset
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self.load_images()

    # Method to load image paths, labels, and filenames
    def load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, filename)
                    # This "label" is wrong
                    label = self.class_to_idx[class_name]
                    # Extract numeric part of the filename (without extension)
                    file_number = int(os.path.splitext(filename)[0])
                    images.append((img_path, label, file_number))
        return images

    # Method to get the length of the dataset
    def __len__(self):
        return len(self.images)

    # Method to get a specific item from the dataset
    # retuns image, label, and filename
    def __getitem__(self, idx):
        img_path, _, filename = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Extract numeric part(in other words, label) of the filename from img_path
        result = re.search(r'/(\d+)/', img_path)
        if result:
            label= int(result.group(1))
        else:
            ValueError("No number found in string")

        return image, label, filename
    

class CustomTestImageDataset(Dataset):
    # Initialization method to set up the dataset
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = self.load_images()

    # Method to load image paths and filenames for test data
    def load_images(self):
        images = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.root_dir, filename)
                images.append((img_path, filename))
        return images

    # Method to get the length of the dataset
    def __len__(self):
        return len(self.images)

    # Method to get a specific item from the dataset
    # returns image, label (set to -1), and filename (set to file name's numeric part)
    def __getitem__(self, idx):
        img_path, filename = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # Extract numeric part (in other words, label) of the filename
        result = re.search(r'\d+', os.path.splitext(filename)[0])
        if result:
            label = -1  # Set label to -1
            filename = int(result.group())
        else:
            raise ValueError("No number found in string")

        return image, label, filename
    
    
def train_loop(model, train_loader, val_loader, criterion, optimizer, n_epochs, device):
    
    loss_history_train = []
    loss_history_val = []
    for epoch in range(n_epochs):
        model.train()
        for inputs, labels, _ in train_loader:
            ###### Your code starts here. ######
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # do forward-backward propogation, and update the model
            ###### Your code starts here. ######
        loss_history_train.append(loss)
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                ###### Your code starts here. ######
                # compute the validation loss and accuracy
                val_loss += criterion(outputs, labels)
                total += labels.size(0)
                predicted = torch.max(outputs.data, 1)[1]
                ###### Your code starts here. ######
                correct += (predicted == labels.squeeze()).sum().item()
            loss_history_val.append(val_loss)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {val_loss/len(val_loader):.4f}, Accuracy: {correct/total:.4f}')

    return loss_history_train, loss_history_val

class TrainModel:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.loss_history_train = []
        self.loss_history_val = []
        self.accuracy_history_train = []
        self.accuracy_history_val = []

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

        average_loss = total_loss / len(loader)
        accuracy = correct / total
        return average_loss, accuracy

    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels, _ in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()

        average_loss = total_loss / len(loader)
        accuracy = correct / total
        return average_loss, accuracy

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            train_loss, train_accuracy = self.train_epoch(self.train_loader)
            val_loss, val_accuracy = self.validate_epoch(self.val_loader)

            self.loss_history_train.append(train_loss)
            self.accuracy_history_train.append(train_accuracy)
            self.loss_history_val.append(val_loss)
            self.accuracy_history_val.append(val_accuracy)

            print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        return {
            'epoch': n_epochs,
            'train_loss': self.loss_history_train,
            'train_accuracy': self.accuracy_history_train,
            'val_loss': self.loss_history_val,
            'val_accuracy': self.accuracy_history_val
        }



def test_loop(model, test_loader, device):
    model = model.to(device)
    y_prob = pd.DataFrame(columns=['ID'])
    model.eval() 
    with torch.no_grad():
        for inputs, _, file_name in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Get the probability
            prob = F.softmax(outputs, dim=1).cpu().numpy()
            prob = pd.DataFrame(prob)
            df = pd.DataFrame(file_name, columns=['ID'])
            # concat the probability and file name
            df = pd.concat([df, prob], axis=1)
            y_prob = pd.concat([y_prob, df], ignore_index=True)
    y_prob= y_prob.sort_values('ID').reset_index(drop=True)

    y_pred = pd.DataFrame(columns=["ID"])
    y_pred["Label"] = y_prob.iloc[:, 1:].idxmax(axis=1)

    return y_prob, y_pred