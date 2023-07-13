import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()



loss = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]*2 
loss = [i+5 for i in loss]
loss2 = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512]




epoch = 10 

for e in range(epoch): 

    writer.add_scalar("loss x epoch", loss[e], e)
    writer.add_scalar("loss2 x epoch", loss2[e], e)
writer.close()




