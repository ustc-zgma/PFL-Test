import sys
import time
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from vgg import VGG_16
from utils import load_imagenet, flatten_paras, ada_pruning

def main():
    print("python version:",sys.version)
    print("torch version:", torch.__version__)
    print("cuda version:", torch.version.cuda)
    model = VGG_16().to('cuda:0')
    summary(model, (3,224,224))
    opt = optim.SGD(model.parameters(), lr=0.0001,weight_decay=5e-4)

    bs = 64
    repeat = 100
    total_epoch_time = 0.0
    total_prun_time = 0.0

    random.seed(0)
    train_data, train_label, _, _ = load_imagenet(1.0, [0,1,2,3,4,5,6,7,8,9], 1)
    indices = [i for i in range(len(train_data[0]))]
    total_round = int (len(indices) / bs)

    model.train()
    for i in range(repeat):	
        random.shuffle(indices)
        epoch_start_time = time.time()
        for round in range(total_round):
            opt.zero_grad()
            start_idx = round * bs
            end_idx = (round+1) * bs
            batch_data = torch.reshape(train_data[0][indices[start_idx:end_idx]], [-1, 3, 224, 224]).to('cuda:0')
            batch_label = (train_label[0][indices[start_idx:end_idx]]).to('cuda:0')
            
            batch_output = model(batch_data)
            batch_loss = F.nll_loss(batch_output, batch_label)
            batch_loss.backward()
            opt.step()
            if round == total_round - 1:
                grads = model.get_grads()
                flatten_grads = torch.abs(flatten_paras(grads))
                value, idx = torch.sort(flatten_grads, descending=True)
                numpy_value = value.cpu().numpy()
                numpy_idx = idx.cpu().numpy()
        prun_start_time = time.time()
        _ = ada_pruning(numpy_value, numpy_idx, 0.7*torch.norm(flatten_paras(grads)), 0.7)
        prun_end_time = time.time()
        epoch_end_time = time.time()
        # print('Epoch [%d]: %.4f / %.4f'%(i, prun_end_time-prun_start_time,epoch_end_time-epoch_start_time))
        total_epoch_time += epoch_end_time-epoch_start_time
        total_prun_time += prun_end_time-prun_start_time
    print("Average used time: pruning time=%.4f s, local updating time=%.4f s"%(total_prun_time/repeat, total_epoch_time/repeat))
main()
