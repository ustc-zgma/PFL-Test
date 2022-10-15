import sys
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from fl_dataset import load_imagenet

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
    
def main():
    print("python version:",sys.version)
    print("torch version:", torch.__version__)
    print("cuda version:", torch.version.cuda)
    device = 'cuda:0'
    model = VGG_16().to(device)

    bs = 64
    random.seed(0)
    train_data, train_label, _, _ = load_imagenet(1.0, [0,1,2,3,4,5,6,7,8,9], 1)
    train_data1, train_label1, _, _ = load_imagenet(1.0, [10,11,12,13,14,15,16,17,18,19], 1)
    print('Load dataset complete..........')
    indices = [i for i in range(len(train_data[0]))]
    # total_round = int (len(indices) / bs)
    total_round = 100
    summary(model, (3,224,224), device=device)
    model.train()
    save_loss = []
    save_loss1 = []
    start_time = time.time()
    random.shuffle(indices)
    for round in range(total_round):
        start_idx = round * bs
        end_idx = (round+1) * bs
        batch_data = torch.reshape(train_data[0][indices[start_idx:end_idx]], [-1, 3, 224, 224]).to(device)
        batch_label = (train_label[0][indices[start_idx:end_idx]]).to(device)
        
        batch_output = model(batch_data)
        batch_loss = F.nll_loss(batch_output, batch_label)
        save_loss.append(batch_loss.item())

        batch_data1 = torch.reshape(train_data1[0][indices[start_idx:end_idx]], [-1, 3, 224, 224]).to(device)
        batch_label1 = (train_label1[0][indices[start_idx:end_idx]]).to(device)
        batch_output1 = model(batch_data1)
        batch_loss1 = F.nll_loss(batch_output1, batch_label1)
        # print('round [%d]: %.2f, %.2f'%(round, batch_loss.item(), batch_loss1.item()))
        save_loss1.append(abs(batch_loss1.item() - batch_loss.item()))
    end_time = time.time()
    avg_time = (end_time-start_time) / total_round
    print("Average update time time=%.4f s, std=%.4f, avg_gap=%.4f"%(avg_time/2, np.std(save_loss), np.mean(save_loss1)))
main()
