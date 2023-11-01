Emotion preditction models based on different types of architectures

Datasets - IEMOCAP dataset, Ravdes Dataset

CNN_3963_3315 model -->

    Training Accuracy - 39.63%
    Validation Accuracy - 33.15%
    
    Architecture

        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super(SimpleCNN, self).__init__()
                
                # Convolutional layers
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) # output - 128x1034
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)# 64x517
                
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1) # 64x517
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(kernel_size=(2,3), stride=(2,4))# 32 x 129
                
                self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1) # 32 x 129
                self.relu3 = nn.ReLU()
                self.pool3 = nn.MaxPool2d(kernel_size=(2,3), stride=2)# 16x64
                
                self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1) # 16x64
                self.relu4 = nn.ReLU()
                self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)# 8x32
        
                # Fully connected layers
                self.fc1 = nn.Linear(32 * 8 * 32, 1024)
                self.relu5 = nn.ReLU()
                self.fc2 = nn.Linear(1024, 128)
                self.relu6 = nn.ReLU()
                self.fc3 = nn.Linear(128, num_classes)

            def forward(self, x):
                #print(x.shape)
                #print(x)
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.pool1(x)
                
                x = self.conv2(x)
                x = self.relu2(x)
                x = self.pool2(x)
                
                x = self.conv3(x)
                x = self.relu3(x)
                x = self.pool3(x)