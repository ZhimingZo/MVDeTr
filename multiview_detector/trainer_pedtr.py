import time
import tqdm
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
import torch.distributed as dist

class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PedTRTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, logdir, dataloader_train, sampler_train, dataloader_test, scheduler, args):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.logdir = logdir
        self.criterion = criterion
        self.optimizer = optimizer
       
        self.epochs=args.epochs
        self.dataloader_train = dataloader_train
        self.sampler_train = sampler_train

        self.dataloader_test = dataloader_test
        self.scheduler = scheduler
        self.log_interval = args.log_interval #100
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.device = args.device
        self.clip_max_norm = args.clip_max_norm
        self.distributed = args.distributed
        #print(self.model, self.criterion, self.optimizer, self.dataloader_train, self.dataloader_test, self.scheduler, self.device)
        #exit()

    def train(self,):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0

        best_moda = 0

        for epoch in tqdm.tqdm(range(1, self.epochs + 1)):
            if self.distributed:
                self.sampler_train.set_epoch(epoch)
            print('Training...:' + str(epoch) +"  LR: " + str(self.scheduler.get_last_lr()))
            loss_epo_box, loss_epo_cls=0, 0 
            
            for batch_idx, (imgs, proj_mats, targets, frame) in enumerate(self.dataloader_train):
                print(str(batch_idx) + ":")
                imgs = imgs.to(self.device)
                proj_mats=proj_mats.to(self.device)
                targets = [{k: v.to(self.device).squeeze() for k, v in targets.items()}]

                # supervised
                outputs  = self.model(img=imgs, proj_mat=proj_mats)
                loss = 0
                loss_dict_list = []
                for output in outputs: 
                    loss_dict = self.criterion(output, targets)
                    weight_dict = self.criterion.weight_dict
                    loss += sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                    loss_dict_list.append(loss_dict)
                
                if dist.get_rank() == 0:
                    loss_epo_box += loss_dict_list[-1]['loss_bbox']
                    loss_epo_cls += loss_dict_list[-1]['loss_ce']
                    print('boxes loss: ' + str(loss_dict_list[-1]['loss_bbox']))
                    print('class loss: ' + str(loss_dict_list[-1]['loss_ce']))
                # multiview regularization
                # Match loss 
                t_f = time.time()
                t_forward += t_f - t_b

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
                self.optimizer.step()
                
                losses += loss.item()
                t_b = time.time()
                t_backward += t_b - t_f
                if (batch_idx + 1) % self.log_interval == 0 or batch_idx + 1 == len(self.dataloader_train):
                    self.logdir.flush() 
            #self.scheduler.step()
            if epoch % 25 == 0: 
                res_fpath = os.path.join(self.logdir, "pred_" + str(epoch)+".txt")
                _, moda = self.test(res_fpath=res_fpath, visualize=False)
                if dist.get_rank() == 0 and moda > best_moda:
                    best_moda = moda
                    torch.save(self.model.state_dict(), os.path.join(self.logdir, 'MultiviewDetector_' + str(epoch)+'.pth'))
                self.model.train()
            if dist.get_rank() == 0:
                print(f'Train Epoch: {epoch}, BboxLoss: {loss_epo_box:.6f}, ClsLoss:{loss_epo_cls:.6f}')
             
        self.logdir.close() 
        return losses / len(self.dataloader_train)
  
    def test(self, res_fpath=None, visualize=False):
        self.model.eval()
        self.criterion.eval()
        losses = 0
        res_list = []
        t0 = time.time()
        print("Evaluating...")
        for batch_idx, (imgs, proj_mats, targets, frame) in enumerate(self.dataloader_test):
            imgs = imgs.to(self.device)
            proj_mats=proj_mats.to(self.device)
            targets = [{k: v.to(self.device).squeeze() for k, v in targets.items()}]
            # with autocast():
            with torch.no_grad():
                #print(imgs.shape)
                print(frame)

                outputs  = self.model.module(img=imgs, proj_mat=proj_mats)[-1]
                probas = F.softmax(outputs['pred_logits'], -1)[0]
                #keep = probas.max(-1).values #> 0.7 #
                #score, index = probas.max(-1)#.values #> 0.7 #
                #print(probas)
                #exit()
            
                index = torch.nonzero((probas[..., 1] > 0.7).to(torch.int32)).flatten()
                score = probas[index, 1] 
                #print(index.shape, score.shape)
                #exit()
                #print(outputs['pred_boxes'].shape) # [1, 100, 2]
                #print(keep.shape) # [1, 100]
                #exit()
                boxes = outputs['pred_boxes'][0]
                boxes = boxes[index]
                boxes[:, 0] = (boxes[:, 0] * self.dataloader_test.dataset.world_grid_shape[0]).long()
                boxes[:, 1] = (boxes[:, 1] * self.dataloader_test.dataset.world_grid_shape[1]).long()
                res = boxes.cpu()
                res = torch.concat((torch.ones(boxes.shape[0]).unsqueeze(1)*frame.cpu(), res), axis=1)
                res_list.append(res)
                
                #res = res.cpu().numpy() 
                #res = np.
                
            #print(boxes.shape)
            #print(boxes)
            #print(frame)
            #exit()
            '''
            if res_fpath is not None:
                xys = mvdet_decode(torch.sigmoid(world_heatmap.detach().cpu()), world_offset.detach().cpu(),
                                   reduce=self.dataloader.dataset.world_reduce)
                # xys = mvdet_decode(world_heatmap.detach().cpu(), reduce=dataloader.dataset.world_reduce)
                grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
                if self.dataloader.dataset.indexing == 'xy':
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
            '''
        t1 = time.time()
        t_epoch = t1 - t0
  
        '''
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
        '''
        if res_fpath is not None:
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')
            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath),
                                                     os.path.abspath(self.dataloader_test.dataset.gt_fpath),
                                                     self.dataloader_test.dataset)
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%')
        else:
            moda = 0
        print(f'Test, Time: {t_epoch:.3f}')
        return losses / len(self.dataloader_test), moda
