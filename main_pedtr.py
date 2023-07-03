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
from torch.cuda.amp import GradScaler
from torch import optim
from torch.utils.data import DataLoader
from multiview_detector.datasets_pedtr.ped_dataset import build_dataset
#from multiview_detector.models.pedtr.PedTransformer import build_model
from multiview_detector.models.pedtr.detectors.pedtr import build_model
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.trainer_pedtr import PedTrainer
from torchvision import transforms as T
import warnings 
warnings.filterwarnings("ignore")


 
def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
    else:
        print('No sys.gettrace')
        is_debug = False

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

     
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    train_set = build_dataset(isTrain=True, args=args)
    test_set = build_dataset(isTrain=False, args=args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)
     
    # logging
    if args.resume is None:
        logdir = f'logs/{args.dataset}/{"debug_" if is_debug else ""}' \
                 f'lr{args.lr}_baseR{args.base_lr_ratio}_' \
                 f'drop{args.dropout}_dropcam{args.dropcam}_' \
                 f'worldR{args.world_grid_reduce}_imgR{args.img_reduce}_' \
                 f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
        os.makedirs(logdir, exist_ok=True)
       
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    else:
        #logdir = f'logs_pretrained/{args.dataset}/{args.resume}'
        logdir = f'logs/{args.dataset}/{args.resume}'
        #logdir = f'{args.resume}'
    print(logdir)
    print('Settings:')
    print(vars(args))
    # model
     
    model, criterion = build_model(args)
    #print(model)
    #print(criterion)
    #exit()
    param_dicts = [{"params": [p for n, p in model.named_parameters() if 'img_backbone' not in n and p.requires_grad], "lr": args.lr},
                   {"params": [p for n, p in model.named_parameters() if 'img_backbone' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, }, ]
    
    optimizer = optim.AdamW(param_dicts, weight_decay=args.weight_decay)
     
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
    #                                                epochs=args.epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)

    trainer = PedTrainer(model=model, optimizer=optimizer, criterion=criterion, logdir=logdir, dataloader_train=train_loader, dataloader_test=test_loader, scheduler=scheduler, args=args)

    # draw curve
    x_epoch = []
    train_loss_s = []
    test_loss_s = []
    test_moda_s = []
    # learn
    res_fpath = os.path.join(logdir, 'test.txt')
    if args.resume is None:
            train_loss = trainer.train()
            #print('Testing...')
            #test_loss, moda = trainer.test(epoch, test_loader, res_fpath, visualize=True)

            # draw & save
            #x_epoch.append(epoch)
            ##train_loss_s.append(train_loss)
            #test_loss_s.append(test_loss)
            #test_moda_s.append(moda)
            #draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s, test_moda_s)
            torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
    else:
        #model.load_state_dict(torch.load(f'logs_pretrained/{args.dataset}/{args.resume}/MultiviewDetector.pth'))
        model.load_state_dict(torch.load(f'logs/{args.dataset}/{args.resume}/MultiviewDetector.pth'))
        #model.load_state_dict(torch.load(logdir))
        model.eval()
    print('Test loaded model...')
    trainer.test(res_fpath, visualize=False)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('--dropcam', type=float, default=0) # org 0 
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') 
    #parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--base_lr_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    #parser.add_argument('--augmentation', type=str2bool, default=True)

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

    # Model 
    #parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'mobilenet'])
    parser.add_argument('--dropout', type=float, default=0.1)
    #parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--embed_dims', type=int, default=512)

    # Matcher 
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost") 
    
    # Loss coefficients 
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--ce_loss_coef', default=1, type=float) # org_1
    parser.add_argument('--eos_coef', default=0.001, type=float, 
                        help="Relative classification weight of the no-object class") # org_0.1

    args = parser.parse_args()
     
     
    main(args)
