from torch.nn import Module, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Softmax
import torch.nn.functional as F
import torch.nn as nn

class ClassifierCNN(nn.Module):
    def __init__(self):
        super(ClassifierCNN, self).__init__()
        self.conv1 = Conv2d(in_channels= 3, out_channels = 16, kernel_size= 3, padding = 1 ) # Input: 3 x 128 x 128, Output: 16 x 128 x 128
        self.pool = MaxPool2d(kernel_size = 2, stride= 1)  # Output 1: 16 x 64 x 64 Output 2: 32 x 32 x 32 Output 3: 64 x 16 x 16
        self.conv2 = Conv2d(in_channels= 16, out_channels = 32, kernel_size= 3, padding = 1 ) # Output: 32 x 32 x 32
        self.conv3 = Conv2d(in_channels= 32, out_channels = 64, kernel_size= 3, padding = 1 ) # Output: 64 x 16 x 16
        self.flatten = Flatten()
        self.fc1 = Linear(in_features= 64 * 16 * 16, out_features= 128)
        self.fc2 = Linear(in_features= 128, out_features= 2)
        self.softmax = Softmax(dim=1) 
        self.relu = ReLU()



    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x