import torch 
import torch.nn as nn 
import torch.nn.functional as F
from multiview_detector.models.pedtr.utils.query_generator_copy import Query_generator
from multiview_detector.models.pedtr.backbones.resnet import resnet18, resnet34, resnet50
from multiview_detector.models.pedtr.dense_heads.pedtr_head import PedTRHead
from multiview_detector.loss.pedtr.matcher_focal_loss import * 
from multiview_detector.loss.pedtr.criterion_focal_loss import SetCriterion_focal
from multiview_detector.loss.pedtr.matcher import * 
from multiview_detector.loss.pedtr.criterion import SetCriterion_ce

class PedTR(nn.Module):
    def __init__(self, args):
        super(PedTR, self).__init__()  

        self.num_query = args.num_queries
        self.embed_dims = args.embed_dims
        self.feat_reduce = args.feat_reduce
        self.upsample_shape = list(map(lambda x: int(x / self.feat_reduce), args.org_img_shape))
        
        #self.backbone = nn.Sequential(*list(resnet18(pretrained=True,
        #                                             replace_stride_with_dilation=[False, True, True]).children())[:-2])
        self.backbone = nn.Sequential(*list(resnet18(pretrained=True,
                                                    replace_stride_with_dilation=[False, False, False]).children())[:-2])


        #self.reduced_features = nn.Sequential(nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1))

        #self.query_generator = Query_generator(args=args, num_query=self.num_query, dims=self.embed_dims)
        self.query_generator = Query_generator(img_backbone=self.backbone, num_query=self.num_query, dims=self.embed_dims)
        self.detect_head = PedTRHead(args)

    def forward(self, org_img, cam_rays=None, proj_mat=None): 
        # extract image features 
        assert len(org_img.shape) == 5
        B, N, C, H, W = org_img.shape
        img = org_img.reshape(B*N, C, H, W) #7, 3, 1080, 1920 
        img_features = self.backbone(img)  #  torch.Size([7, 512, 90, 160]) after resize 
        #img_features = self.reduced_features(img_features)
        
        # up_sample feature map 
        if  self.feat_reduce != 1:
            img_features = F.interpolate(img_features, self.upsample_shape, mode='bilinear')

        # generate query (either view & ray encoded or normal query) and learnable positional embedding 
        #query, query_pos = self.query_generator(img_feat=img_features, ray=cam_rays) # torch.Size([100, 512]) torch.Size([100, 512])  
        query, query_pos = self.query_generator(img=org_img)      
        # output the final output from detect head 
        # out:[{output_class, output_coords} X number of decoder_layers] 
        out = self.detect_head(img_features=img_features, proj_mats=proj_mat, query=query, query_pos=query_pos)

        return  out 




def build_model(args): 
    device = torch.device(args.device)
    # build_model  
    model = PedTR(args).to(device)
     
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    losses = ['labels', 'boxes']
    # build matcher & criterion
    if args.loss == 'ce':
        matcher = build_matcher_ce(args)
        criterion = SetCriterion_ce(num_classes=2, matcher=matcher, weight_dict=weight_dict,
                            losses=losses, eos_coef=args.eos_coef)
    else:
        matcher = build_matcher_focal(args)
        criterion = SetCriterion_focal(num_classes=2, matcher=matcher, weight_dict=weight_dict,
                            losses=losses)
    criterion.to(device)
    return model, criterion




def test(): 
    #tensor = torch.rand((1, 7, 3, 1080, 1920))
    #model = PedTR()
    #print(model(tensor))
    proj_mats = torch.rand((1, 7, 3, 3))
    imgs = torch.rand((1, 7, 3, 1080, 1920))
    PedTRModel = PedTR()
    out = PedTRModel(imgs=imgs, proj_mats=proj_mats) 

    out_coord =  out['pred_boxes']
    out_cls = out['pred_logits']
    print(out_coord.shape, out_cls.shape)
    pass    
#test()











'''
from torchvision import transforms as T
img = T.ToPILImage()(img[0].squeeze().cpu())
img.save("test_.png")
exit()
''' 