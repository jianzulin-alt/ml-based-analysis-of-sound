"""
CNN from "Detecting and Classifying Musical Instruments with
Convolutional Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioSpectrogramCNN(nn.Module):
    """
    Input: (B, 2, 128, W) 2-channel mel-spectrogram.
    """

    def __init__(self, in_ch: int = 2, num_classes: int = 11, p_drop: float = 0.5):
        super().__init__()

        # Conv1: 4x4 kernel, 8 filters 
        self.conv1 = nn.Conv2d(in_ch, 8, kernel_size=4, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv2: 3x3 kernel, 16 filters 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Conv3: 2x2 kernel, 32 filters 
        self.conv3 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Conv4: 12x2 kernel, 64 filters 
        # Padding adjusted to handle the large asymmetric kernel without collapsing spatial dims too early
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(12, 2), stride=1, padding=(5, 1))
        self.bn4   = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.drop1 = nn.Dropout(p_drop)
        
        # Using LazyLinear to automatically calculate the flattened shape
        # The paper specifies 500 hidden units for the first FC layer 
        self.fc1   = nn.LazyLinear(500) 
        self.drop2 = nn.Dropout(p_drop)
        
        # Final output layer (11 classes)
        self.fc2   = nn.Linear(500, num_classes)

    @staticmethod
    def _act(x):
        # The paper explicitly uses ReLU activations, not SiLU 
        return F.relu(x)

    def _forward_features(self, x):
        x = self.pool1(self._act(self.bn1(self.conv1(x))))
        x = self.pool2(self._act(self.bn2(self.conv2(x))))
        x = self.pool3(self._act(self.bn3(self.conv3(x))))
        x = self.pool4(self._act(self.bn4(self.conv4(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)   
        
        
        x = torch.flatten(x, 1)         
        
        x = self.drop1(x)
        x = self._act(self.fc1(x))      # (B, 500)
        x = self.drop2(x)
        
        # Raw logits output. 
        return self.fc2(x)