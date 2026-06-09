import torch
import torch.nn as nn
import torch.nn.functional as F
import src.config as config

class CRNN(nn.Module):
    def __init__(self, num_classes=37):
        super(CRNN, self).__init__()
        
        # ----------------------------------------------------------------------
        # Phase 1: CNN Feature Extractor
        # ----------------------------------------------------------------------
        # Input shape: [Batch, 1, 32, 128]
        
        self.conv1 = nn.Conv2d(config.IMG_CHANNELS, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2) # Output: [Batch, 64, 16, 64]
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2) # Output: [Batch, 128, 8, 32]
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # Critical shift: Pool height by 2, but keep width by 1. 
        # We need to preserve the horizontal dimension for our sequence time-steps.
        self.pool3 = nn.MaxPool2d((2, 1)) # Output: [Batch, 256, 4, 32]
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1)) # Output: [Batch, 256, 2, 32]
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d((2, 1)) # Output: [Batch, 512, 1, 32]

        # ----------------------------------------------------------------------
        # Phase 2: Sequence Modeling (BiLSTM)
        # ----------------------------------------------------------------------
        # The CNN gives us 512 channels. We want the LSTM to read these as features.
        self.lstm_hidden = 256
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=self.lstm_hidden, 
            bidirectional=True, 
            num_layers=2,
            dropout=0.25 # Slight regularization to prevent memorizing the static noise
        )
        
        # ----------------------------------------------------------------------
        # Phase 3: Classifier
        # ----------------------------------------------------------------------
        # BiLSTM concatenates forward and backward passes, so hidden size is doubled (256 * 2)
        self.fc = nn.Linear(self.lstm_hidden * 2, num_classes)

    def forward(self, x):
        # Push through CNN blocks with ReLU activations
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        
        # At this point, x is shape [Batch, Channels=512, Height=1, Width=32]
        
        # Squeeze out the height dimension since it's 1
        x = x.squeeze(2) # Now [Batch, 512, 32]
        
        # LSTMs expect the time sequence to be the very first dimension: [Time, Batch, Features]
        # We permute to swap the Batch (0) and Width/Time (2) axes.
        x = x.permute(2, 0, 1) # Now [32, Batch, 512]
        
        # Pass through the BiLSTM
        x, _ = self.lstm(x) # Output is still [32, Batch, 512] because of bidirectional concat
        
        # Map features to our vocabulary space
        x = self.fc(x) # Now [32, Batch, 37]
        
        # CTC Loss expects log probabilities, so we apply log_softmax over the class dimension
        return F.log_softmax(x, dim=2)