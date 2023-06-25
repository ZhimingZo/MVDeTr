import time
import tqdm
import os
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from PIL import Image
from multiview_detector.loss import *
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image, img_color_denormalize
import math
import sys

class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PedTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, logdir, dataloader_train, dataloader_test, scheduler, args):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.logdir = logdir
        self.criterion = criterion
        self.optimizer = optimizer

        self.epochs=args.epochs
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.scheduler = scheduler
        self.log_interval = args.log_interval
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.device = args.device


    def train(self,):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for epoch in tqdm.tqdm(range(1, self.epochs + 1)):
            print('Training...:' + str(epoch))
            for batch_idx, (imgs, proj_mats, targets, frame) in enumerate(self.dataloader_train):
                imgs = imgs.to(self.device)
                proj_mats=proj_mats.to(self.device)
                targets = [{k: v.to(self.device).squeeze() for k, v in targets.items()}]
                
                # supervised
                outputs  = self.model(img=imgs, proj_mat=proj_mats)
                 
                loss_dict = self.criterion(outputs, targets)
                weight_dict = self.criterion.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
                # multiview regularization
                # Match loss 
                t_f = time.time()
                t_forward += t_f - t_b

                self.optimizer.zero_grad()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                self.optimizer.step()
                losses += loss.item()
                print(loss.item())
                if not math.isfinite(loss):
                    print("Loss is {}, stopping training".format(loss))
                    print(loss)
                    sys.exit(1)

               
                
                
                t_b = time.time()
                t_backward += t_b - t_f

               
                self.scheduler.step()
                if (batch_idx + 1) % self.log_interval == 0 or batch_idx + 1 == len(self.dataloader_train):
                    t1 = time.time()
                    t_epoch = t1 - t0
                    print(f'Train Epoch: {epoch}, Batch:{(batch_idx + 1)}, loss: {losses / (batch_idx + 1):.6f}, '
                        f'Time: {t_epoch:.1f}')
                #print("hello")
        return losses / len(self.dataloader_train)
'''   
    def test(self,res_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        res_list = []
        t0 = time.time()
        for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame) in enumerate(self.dataloader_test):
            B, N = imgs_gt['heatmap'].shape[:2]
            data = data.cuda()
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
            # with autocast():
            with torch.no_grad():
                output_coords, out_cls  = self.model(img=imgs, proj_mat=proj_mats)

            if res_fpath is not None:
                xys = mvdet_decode(torch.sigmoid(world_heatmap.detach().cpu()), world_offset.detach().cpu(),
                                   reduce=dataloader.dataset.world_reduce)
                # xys = mvdet_decode(world_heatmap.detach().cpu(), reduce=dataloader.dataset.world_reduce)
                grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
                if dataloader.dataset.base.indexing == 'xy':
                    positions = grid_xy
                else:
                    positions = grid_xy[:, :, [1, 0]]

                for b in range(B):
                    ids = scores[b].squeeze() > self.cls_thres
                    pos, s = positions[b, ids], scores[b, ids, 0]
                    res = torch.cat([torch.ones([len(s), 1]) * frame[b], pos], dim=1)
                    ids, count = nms(pos, s, 20, np.inf)
                    res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                    res_list.append(res)
        t1 = time.time()
        t_epoch = t1 - t0
  
        
        if visualize:
            # visualizing the heatmap for world
            fig = plt.figure()
            subplt0 = fig.add_subplot(211, title="output")
            subplt1 = fig.add_subplot(212, title="target")
            subplt0.imshow(world_heatmap.cpu().detach().numpy().squeeze())
            subplt1.imshow(world_gt['heatmap'].squeeze())
            plt.savefig(os.path.join(self.logdir, f'world{epoch if epoch else ""}.jpg'))
            plt.close(fig)
            # visualizing the heatmap for per-view estimation
            heatmap0_foot = imgs_heatmap[0].detach().cpu().numpy().squeeze()
            img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
            img0 = Image.fromarray((img0 * 255).astype('uint8'))
            foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
            foot_cam_result.save(os.path.join(self.logdir, 'cam1_foot.jpg'))
        
        if res_fpath is not None:
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')
            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath),
                                                     os.path.abspath(dataloader.dataset.gt_fpath),
                                                     dataloader.dataset.base.__name__)
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%')
        else:
            moda = 0
        #print(f'Test, loss: {losses / len(dataloader):.6f}, Time: {t_epoch:.3f}')
        print(f'Test, Time: {t_epoch:.3f}')
        return losses / len(dataloader), moda
'''       