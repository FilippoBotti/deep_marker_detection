import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 2)  # Output layer with one unit for regression
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    # def __init__(self):
    #     super(CNNModel, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
    #     self.norm1 = nn.BatchNorm2d(16)
    #     self.relu1 = nn.ReLU()
    #     self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.first_layer = nn.Sequential(self.conv1, self.norm1, self.relu1, self.pool1)
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    #     self.norm2 = nn.BatchNorm2d(32)
    #     self.relu2 = nn.ReLU()
    #     self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.second_layer = nn.Sequential(self.conv2, self.norm2, self.relu2, self.pool2)
    #     self.flatten = nn.Flatten()
    #     self.fc1 = nn.Linear(32 * 8 * 8, 128)
    #     self.relu3 = nn.ReLU()
    #     self.fc2 = nn.Linear(128, 2)  # Output layer with 2 neurons for (x, y) prediction
    #     self.fc = nn.Sequential(self.flatten, self.fc1, self.fc2)
    #     # self.model = models.resnet50()
    #     # self.model.fc = torch.nn.Sequential(
    #     #         torch.nn.Linear(
    #     #             in_features=2048,
    #     #             out_features=2
    #     #         ),
    #     #     )
    # def forward(self, x):
    #     x = self.first_layer(x)
    #     x = self.second_layer(x)
    #     x = self.fc(x)
    #     return x

class FC2Layer(nn.Module):
    def __init__(self, input_size, n_hidden, output_size):
        super(FC2Layer, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, n_hidden), 
            nn.ReLU(), 
            nn.Linear(n_hidden, n_hidden), 
            nn.ReLU(), 
            nn.Linear(n_hidden, output_size), 
        )

    def forward(self, x):
        x = x.view(-1, self.input_size) # flatten
        return self.network(x)