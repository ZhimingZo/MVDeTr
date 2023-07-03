import torch 
import torch.nn as nn 
from multiview_detector.models.pedtr.utils.query_generator import Query_genrator
from multiview_detector.models.pedtr.backbones.resnet import resnet18
from multiview_detector.models.pedtr.dense_heads.pedtr_head import PedTRHead
from multiview_detector.loss.pedtr.matcher import * 
from multiview_detector.loss.pedtr.criterion import SetCriterion

class PedTR(nn.Module):
    def __init__(self, ):
        super(PedTR, self).__init__()  

        self.backbone = nn.Sequential(*list(resnet18(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[:-2])
        self.query_generator = Query_genrator(num_query=100, dims=512)
  
        self.detect_head = PedTRHead(num_decoder_layer=6)

    def forward(self, img, cam_rays=None, proj_mat=None): 
        # extract image features 
        assert len(img.shape) == 5
        B, N, C, H, W = img.shape
        img = img.reshape(B*N, C, H, W) #7, 3, 1080, 1920 
        '''
        from torchvision import transforms as T
        img = T.ToPILImage()(img[0].squeeze().cpu())
        img.save("test_.png")
        exit()
        ''' 
        img_features = self.backbone(img)  # torch.Size([7, 512, 135, 240])

        # generate query (either view & ray encoded or normal query) and learnable positional embedding 
        query, query_pos = self.query_generator(img, cam_rays) # torch.Size([100, 512]) torch.Size([100, 512])
        
        # output the final output from detect head 
        # out:{output_class, output_coords}
        out = self.detect_head(img_features=img_features, proj_mats=proj_mat, query=query, query_pos=query_pos)
         
        return  out 


def build_model(args): 

    device = torch.device(args.device)
    # build_model  
    model = PedTR().to(device)
    # build matcher 
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    #losses = ['labels', 'boxes']
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes=2, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    #print(model)
    #exit()
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