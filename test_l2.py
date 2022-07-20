import sys
import time
import torch
from torchsummary import summary
from vgg import VGG_16
from utils import flatten_paras

def main():
    print("python version:",sys.version)
    print("torch version:", torch.__version__)
    print("cuda version:", torch.version.cuda)
    model1 = VGG_16().to('cuda:0')
    
    summary(model1, (3,224,224))
    repeat = 100
    start_time = time.time()
    for _ in range(repeat):	
        para1 = model1.get_paras()
        l2_norm = torch.norm(flatten_paras(para1))
		# print(l2_norm.item())
    end_time = time.time()
    total_time = end_time - start_time
    print("Average used time: %.4f s"%(total_time / repeat))
main()
