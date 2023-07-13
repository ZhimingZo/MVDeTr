import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import random
import numpy as np
import torch
#from torch.cuda.amp import GradScaler
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from multiview_detector.datasets_pedtr.ped_dataset import build_dataset
from multiview_detector.models.pedtr.detectors.pedtr import build_model
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.trainer_pedtr import PedTRTrainer
from torchvision import transforms as T
import warnings 
import multiview_detector.utils.misc as utils
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
 
def main(args):


    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True
   
    # logging
     
    if args.resume is None and dist.get_rank()==0:
        logdir = f'logs_focal/{args.dataset}/' \
                f'lr{args.lr}_baseR{args.base_lr_ratio}_' \
                f'drop{args.dropout}_dropcam{args.dropcam}_' \
                f'worldR{args.world_grid_reduce}_imgR{args.img_reduce}_' \
                f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
        os.makedirs(logdir, exist_ok=True)
        source = './multiview_detector'
        destination = logdir + '/scripts/multiview_detector'
        for root, dirs, files in os.walk(source):
            rel_path = os.path.relpath(root, source)
            dest_dir = os.path.join(destination, rel_path)
            os.makedirs(dest_dir, exist_ok=True)
            for script in os.listdir('.'):
               if script.split('.')[-1] == 'py':
                   dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                   shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    else:
        logdir = f'logs_focal/{args.dataset}/{args.resume}'

    print(logdir)
    print('Settings:')
    print(vars(args))


    # loss writer 
    writer = SummaryWriter()
    
    # model
    model, criterion = build_model(args)
    model_without_ddp = model 
    if args.distributed: 
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
         
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if 'backbone' not in n and p.requires_grad], "lr": args.lr},
                   {"params": [p for n, p in model_without_ddp.named_parameters() if 'backbone' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, }, ]
    
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.lr_gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
     # dataset
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset_train = build_dataset(isTrain=True, args=args)
    dataset_test = build_dataset(isTrain=False, args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_test)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, sampler=sampler_val, drop_last=False,  num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)
     
    trainer = PedTRTrainer(model=model, optimizer=optimizer, criterion=criterion, logdir=logdir, dataloader_train=data_loader_train,\
                          sampler_train=sampler_train, dataloader_test=data_loader_test, scheduler=scheduler, args=args, loss_writer=writer)

    # learn
    if args.resume is None:
            train_loss = trainer.train()
    else:
        model_without_ddp.load_state_dict(torch.load(f'logs_focal/{args.dataset}/{args.resume}/MultiviewDetector_best.pth'))
        model_without_ddp.eval()
    res_fpath = os.path.join(logdir, 'best_model.txt')
    print('Test loaded model...')
    trainer.test(res_fpath, visualize=False)
    # clean up 
    if args.distributed:
        dist.destroy_process_group()
if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--dropcam', type=float, default=0) # org 0 
    parser.add_argument('--epochs', type=int, default=31, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate') 
    parser.add_argument('--base_lr_ratio', type=float, default=0.1)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--lr_gamma', default=0.9, type=int)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='random seed') # org 2021
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--clip_max_norm', default=35, type=float,
                        help='gradient clipping max norm')

    #Dataset 
    parser.add_argument('-r', '--root', type=str, default='../Data/')
    parser.add_argument('-d', '--dataset', type=str, default='Wildtrack', choices=['Wildtrack', 'MultiviewX'])
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--world_grid_reduce', type=int, default=1)
    parser.add_argument('--img_reduce', type=int, default=1)
    parser.add_argument('--num_cams', type=int, default=7)   # 6 for MultiviewX
    parser.add_argument('--num_frames', type=int, default=2000) # 400 for MultiviewXgit 
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int) # org 4

    # Model 
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_dims', type=int, default=512)
    parser.add_argument('--num_decoder_layer', type=int, default=6)
    parser.add_argument('--num_queries', default=900, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="Number of heads of MultiHeadAttn")


    # Matcher 
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost") #1 
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost") #5 
    
    # Loss coefficients 
    parser.add_argument('--bbox_loss_coef', default=5, type=float) #5 
    parser.add_argument('--ce_loss_coef', default=1, type=float) #1
    parser.add_argument('--eos_coef', default=0.1, type=float, 
                        help="Relative classification weight of the no-object class") # org_0.1
    


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=1, type=int,
                        help='local rank')


    args = parser.parse_args()
    main(args)
