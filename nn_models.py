import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np



class MLP(nn.Module):
    def __init__(self, in_dim,hidden_dim, out_dim, dropout ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim //2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_dim // 4, out_dim)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.fc4(x)

        return x
    
    def get_latent(self, data_loader, device):
        """
        Get the latent representations of the data using the model.
        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            device (torch.device): Device to run the model on (CPU or GPU).
        Returns:
            x_representations (numpy.ndarray): Latent representations of the data.
            y_labels (numpy.ndarray): Corresponding labels of the data.
        """
        self.eval()
        x_representations = []
        y_labels = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)

                x = self.fc1(inputs)
                x = self.relu1(x)
                x = self.bn1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.bn2(x)
                x = self.fc3(x)
                x = self.relu3(x)
                x = self.bn3(x)

                x_representations.append(x.cpu().numpy())
                y_labels.append(labels.cpu().numpy())

        x_representations = np.concatenate(x_representations, axis=0)
        y_labels = np.concatenate(y_labels, axis=0)
        return x_representations, y_labels


class Resnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0):
        super(Resnet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(hidden_dim//2)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_dim//2, hidden_dim //2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(hidden_dim//2)
        self.dropout4 = nn.Dropout(dropout)
        self.fc5 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm1d(hidden_dim // 4)
        self.dropout5 = nn.Dropout(dropout)
        self.fc6 = nn.Linear(hidden_dim//4, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        i = x
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = i + x
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        i = x
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = x+i
        x = self.dropout4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.dropout5(x)
        x = self.fc6(x)
        

        return x