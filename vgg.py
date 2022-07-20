import torch.nn as nn
import torch.nn.functional as F

class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
     
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(128,256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256,256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256,256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(256,512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512,512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512,512, kernel_size=3, padding=1)

        
        self.conv11 = nn.Conv2d(512,512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(512,512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(512,512, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 100)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x, inplace=True)
        x = self.bn1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x, inplace=True)
        x = self.bn3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x, inplace=True)
        x = self.bn4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x, inplace=True)
        x = self.bn5(x)

        x = x.view(-1,512*7*7)
        F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
    
    def get_paras(self):
        return [p[1].data.clone().detach() for p in self.named_parameters()]
    
    def get_grads(self):
        return [p[1].grad.clone().detach() for p in self.named_parameters()]
