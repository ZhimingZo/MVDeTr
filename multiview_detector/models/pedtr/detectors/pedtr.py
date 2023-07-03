import torch 
import torch.nn as nn 
from multiview_detector.models.pedtr.utils.query_generator import Query_genrator
from multiview_detector.models.pedtr.backbones.resnet import resnet18
from multiview_detector.models.pedtr.dense_heads.pedtr_head import PedTRHead
class PedTR(nn.Module):
    def __init__(self, ):
        super(PedTR, self).__init__()  

        self.backbone = nn.Sequential(*list(resnet18(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[:-2])
        self.query_generator = Query_genrator(num_query=100, dims=512)
  
        self.detect_head = PedTRHead(num_decoder_layer=6)

    def forward(self, imgs, cam_rays=None, proj_mats=None): 
        # extract image features 
        assert len(imgs.shape) == 5
        B, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B*N, C, H, W)

        imgs_features = self.backbone(imgs)  # torch.Size([7, 512, 135, 240])
         
        # generate query (either view & ray encoded or normal query) and learnable positional embedding 
        query, query_pos = self.query_generator(imgs, cam_rays) # torch.Size([100, 512]) torch.Size([100, 512])
        
        # output the final output from detect head 
        # out:{output_class, output_coords}
        out = self.detect_head(img_features=imgs_features, proj_mats=proj_mats, query=query, query_pos=query_pos)
        return  out 



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
test()