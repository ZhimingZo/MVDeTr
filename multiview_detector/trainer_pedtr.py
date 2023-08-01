import time
import tqdm
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
#from PIL import Image
#from multiview_detector.loss import *
from multiview_detector.evaluation.evaluate import evaluate
import torch.distributed as dist
from multiview_detector.utils.nms import nms
from multiview_detector.loss.pedtr.exclusive_matching import exclusive_matching_loss

class PedTRTrainer(object):
    def __init__(self, model, optimizer, criterion, logdir, dataloader_train, sampler_train, dataloader_test, scheduler, args, loss_writer):
        super(PedTRTrainer, self).__init__()
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
        #self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.device = args.device
        self.clip_max_norm = args.clip_max_norm
        self.distributed = args.distributed
        self.world_size = dist.get_world_size() if args.distributed else 1
        self.args = args
        #print(self.model, self.criterion, self.optimizer, self.dataloader_train, self.dataloader_test, self.scheduler, self.device)
        #exit()
        self.loss_writer = loss_writer
    def train(self,):
        self.model.train()
        self.criterion.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        best_moda = -1
        for epoch in tqdm.tqdm(range(1, self.epochs + 1)):
            if self.args.distributed:
                self.sampler_train.set_epoch(epoch)
            print('Training...:' + str(epoch) +"  LR: " + str(self.scheduler.get_last_lr()))
            loss_epo_box, loss_epo_cls, loss_epo_match = 0, 0, 0 
            for batch_idx, (imgs, proj_mats, targets, rays, frame) in enumerate(self.dataloader_train):
                print(str(batch_idx) + ":")
                imgs = imgs.to(self.device)
                proj_mats=proj_mats.to(self.device)
                rays = rays.to(self.device) if self.args.use_rays else None
                targets = [{k: v.to(self.device).squeeze() for k, v in targets.items()}]
                # supervised
                outputs  = self.model(org_img=imgs, proj_mat=proj_mats, cam_rays=rays)
                loss = 0
                loss_dict_list = []
                indices = None
                match_loss = 0
                for output in outputs: 
                    loss_dict, indices  = self.criterion(output, targets)
                    weight_dict = self.criterion.weight_dict
                    match_loss =  exclusive_matching_loss(output['pred_boxes'], indices[0][0]) * 0.1
                    loss += sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) + match_loss
                    loss_dict_list.append(loss_dict)
                if not self.distributed or dist.get_rank() == 0:
                    loss_epo_box += loss_dict_list[-1]['loss_bbox']   
                    loss_epo_cls += loss_dict_list[-1]['loss_ce']  
                    loss_epo_match += match_loss 
                    #loss_epo_giou += loss_dict_list[-1]['loss_giou']  
                    print('boxes loss: ' + str(loss_dict_list[-1]['loss_bbox']))
                    print('class loss: ' + str(loss_dict_list[-1]['loss_ce']))
                    print('exclusive_matching_loss: ' + str(match_loss))
                    #print('giou loss: ' + str(loss_dict_list[-1]['loss_giou']))
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
            if not self.distributed or dist.get_rank() == 0:
                res_fpath = os.path.join(self.logdir, "pred_" + str(epoch)+".txt")
                _, moda, loss_epo_box_val, loss_epo_cls_val = self.test(res_fpath=res_fpath, visualize=False)
                if moda > best_moda:
                    best_moda = moda
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.module.state_dict(),
                        #'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss_box': loss_epo_box * self.world_size,
                        'loss_cls': loss_epo_cls * self.world_size,
                        #'loss_giou': loss_epo_giou * self.world_size,
                        # Add any other relevant information
                    }, os.path.join(self.logdir, 'MultiviewDetector_best'+'.pth'))
                    #torch.save(self.model.state_dict(), os.path.join(self.logdir, 'MultiviewDetector_best.pth'))
                if epoch % 1 == 0: 
                    #torch.save(self.model.module.state_dict(), os.path.join(self.logdir, 'MultiviewDetector_' + str(epoch) + '.pth'))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.module.state_dict(),
                        #'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss_box': loss_epo_box * self.world_size,
                        'loss_cls': loss_epo_cls * self.world_size,
                        #'loss_giou': loss_epo_giou * self.world_size,
                        # Add any other relevant information
                    }, os.path.join(self.logdir, 'MultiviewDetector_' + str(epoch) + '.pth'))
                self.model.train()
                self.criterion.train()
            if not self.distributed or dist.get_rank() == 0:
                print(f'Train Epoch: {epoch}, BboxLoss: {loss_epo_box * self.world_size:.2f}, ClsLoss:{loss_epo_cls * self.world_size:.2f}, \
                      Matching Loss: {loss_epo_match * self.world_size:.4f},')
                self.loss_writer.add_scalar("boxes loss x epoch", loss_epo_box * self.world_size, epoch)
                self.loss_writer.add_scalar("classs loss x epoch", loss_epo_cls * self.world_size, epoch)
                self.loss_writer.add_scalar("match loss x epoch", loss_epo_match * self.world_size, epoch)
                #self.loss_writer.add_scalar("giou x epoch", loss_epo_giou * self.world_size, epoch)
                self.loss_writer.flush()
        if not self.distributed or dist.get_rank() == 0:
            self.loss_writer.close()
        return losses / len(self.dataloader_train)
    @torch.no_grad()
    def test(self, res_fpath=None, visualize=False):
        self.model.eval()
        self.criterion.eval()
        losses = 0
        res_list = []
        #idx_list = []
        t0 = time.time()
        loss_epo_box, loss_epo_cls = 0, 0 
        #idx_fpath =  os.path.join(self.logdir, 'idx_epo_24.txt')

        #samples = []
        print("Evaluating...")
        for batch_idx, (imgs, proj_mats, targets, rays, frame) in enumerate(self.dataloader_train):
            print(frame)
            imgs = imgs.to(self.device)
            proj_mats=proj_mats.to(self.device)
            rays = rays.to(self.device) if self.args.use_rays else None
            targets = [{k: v.to(self.device).squeeze() for k, v in targets.items()}]
            outputs  = self.model(org_img=imgs, proj_mat=proj_mats, cam_rays=rays)[-1]
            '''
            i = 0 
            layers = []
            for output in outputs:
                loss, indices = self.criterion(output, targets)
                #print(indices)
                t2 = torch.argsort(indices[0][1])
                idx = indices[0][0][t2]
                #print(idx)
                layers.append(idx)
            #idx = torch.concat((torch.tensor(frame), idx))
            sample = torch.stack(layers, dim=0)
            samples.append(sample)
            
            #print(indices)
            #exit()
           
            #loss_dict = self.criterion(outputs, targets)
            #weight_dict = self.criterion.weight_dict
            #losses += sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            #loss_epo_box += loss_dict['loss_bbox']   
            #loss_epo_cls += loss_dict['loss_ce']   
            '''
            


            
            if self.args.loss == 'ce':
                probas = F.softmax(outputs['pred_logits'], dim=-1)[0]
                #topk_values, topk_indexes = torch.topk(probas[..., 1], 100, dim=0)
                #index = torch.nonzero((probas[topk_indexes, 1]>=0.1).to(torch.int32)).flatten() # org 0.6 
                #score, labels = F.softmax(outputs['pred_logits'], dim=-1)[..., :-1].max(-1)
                #print(probas)
                #print(score, labels)
                #exit()
                #index = topk_indexes[index]
                index = torch.nonzero((probas[..., 1]>0.5).to(torch.int32)).flatten()
                score = probas[index, 1] 
                boxes = outputs['pred_boxes'][0]
                boxes = boxes[index]
     
                #boxes = boxes[indices[0][0].flatten()]
                
            elif self.args.loss == 'focal':
                probas = F.sigmoid(outputs['pred_logits'])[0]
                #topk_values, topk_indexes = torch.topk(probas[..., 1], 50, dim=0)
                #index = torch.nonzero((probas[topk_indexes, 1] >=0.1).to(torch.int32)).flatten() # org 0.6 
                index = torch.nonzero((probas[..., 1] >= 0).to(torch.int32)).flatten() # org 0.6 
                #index = topk_indexes[index]
                #index = topk_indexes

                score = probas[index, 1]  
                boxes = outputs['pred_boxes'][0]
                boxes = boxes[index]

            boxes[:, 0] = (boxes[:, 0] * self.dataloader_test.dataset.world_grid_shape[0]).long()
            boxes[:, 1] = (boxes[:, 1] * self.dataloader_test.dataset.world_grid_shape[1]).long()
             
            #ids, count = nms(boxes, score, 20, np.inf)
            res = boxes.cpu()
            score = score.cpu()*100
            #score = score[ids[:count]]
            #res = torch.concat((torch.ones([count, 1]) *frame.cpu(), res[ids[:count]]), axis=1)
            #print(score)
            res = torch.concat((torch.ones(boxes.shape[0]).unsqueeze(1)*frame.cpu(), res), axis=1)
            res = torch.concat((res, score.unsqueeze(1)), axis=1)
            res_list.append(res)
            '''
            #t2 = torch.argsort(indices[0][1])
            #idx = torch.concat((indices[0][0][t2], indices[0][1][t2]), axis=0)
            #idx = torch.concat((indices[0][0], indices[0][1]), axis=0)
            #idx = torch.concat((torch.tensor(frame), idx))
            #idx_list.append(idx)
            '''
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
        
        #print(len(samples))
        #exit()
        t1 = time.time()
        t_epoch = t1 - t0
        print("hello_world")
        if res_fpath is not None:
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            '''
            with open(idx_fpath, 'w') as f: 
                for tensor_row in idx_list: 
                    f.write(str(tensor_row.numpy()) + '\n')
                f.close()
            '''
            np.savetxt(res_fpath, res_list, '%d')
            
            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath),
                                                     os.path.abspath(self.dataloader_test.dataset.gt_fpath),
                                                     self.dataloader_test.dataset)
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%')
        else:
            moda = 0
        print(f'Test, Time: {t_epoch:.3f}')      
        return losses / len(self.dataloader_test), moda, loss_epo_box, loss_epo_cls


'''
probas = F.softmax(outputs['pred_logits'], dim=-1)[0]
index = torch.nonzero((probas[..., 1] > 0.6).to(torch.int32)).flatten() 
score = probas[index, 1] 
boxes = outputs['pred_boxes'][0]
boxes = boxes[index]
'''