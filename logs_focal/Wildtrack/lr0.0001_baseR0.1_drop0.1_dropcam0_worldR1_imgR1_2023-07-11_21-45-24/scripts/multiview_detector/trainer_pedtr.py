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
    def __init__(self, model, optimizer, criterion, logdir, dataloader_train, sampler_train, dataloader_test, scheduler, args, loss_writer):
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
        self.loss_writer = loss_writer
    def train(self,):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0

        best_moda = -1

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
                
            self.scheduler.step()
            if epoch % 1 == 0 and dist.get_rank() == 0: 
                res_fpath = os.path.join(self.logdir, "pred_" + str(epoch)+".txt")
                _, moda = self.test(res_fpath=res_fpath, visualize=False)
                if moda > best_moda:
                    best_moda = moda
                    torch.save(self.model.module.state_dict(), os.path.join(self.logdir, 'MultiviewDetector_best.pth'))
                self.model.train()
            if dist.get_rank() == 0:
                print(f'Train Epoch: {epoch}, BboxLoss: {loss_epo_box:.6f}, ClsLoss:{loss_epo_cls:.6f}')
                self.loss_writer.add_scalar("boxes loss x epoch", loss_epo_box, epoch)
                self.loss_writer.add_scalar("classs loss x epoch", loss_epo_cls, epoch)
        self.loss_writer.close()
        return losses / len(self.dataloader_train)
    @torch.no_grad()
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
            outputs  = self.model(img=imgs, proj_mat=proj_mats)[-1]
            #probas = F.softmax(outputs['pred_logits'], -1)[0]
            #index = torch.nonzero((probas[..., 1] > 0.7).to(torch.int32)).flatten()
            probas = F.sigmoid(outputs['pred_logits'])[0]
            topk_values, topk_indexes = torch.topk(probas, 100, dim=0)
            topk_indexes = torch.unique(topk_indexes.flatten())
            #index = torch.nonzero((probas[..., 1] > 0.6).to(torch.int32)).flatten()
            index = torch.nonzero((probas[topk_indexes, 1] > 0.6).to(torch.int32)).flatten()
            index = topk_indexes[index]
            #print(probas.shape)
            #index = torch.nonzero(torch.argmax(probas, dim=1)==1).flatten()
            #print(index.shape)
            #print(index)
            score = probas[index, 0] 
            boxes = outputs['pred_boxes'][0]
            boxes = boxes[index]
            boxes[:, 0] = (boxes[:, 0] * self.dataloader_test.dataset.world_grid_shape[0]).long()
            boxes[:, 1] = (boxes[:, 1] * self.dataloader_test.dataset.world_grid_shape[1]).long()
            res = boxes.cpu()
            res = torch.concat((torch.ones(boxes.shape[0]).unsqueeze(1)*frame.cpu(), res), axis=1)
            res_list.append(res)
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
