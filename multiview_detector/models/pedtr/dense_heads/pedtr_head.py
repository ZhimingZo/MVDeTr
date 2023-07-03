import torch 
import torch.nn as nn 
from multiview_detector.models.pedtr.utils.pedtr_transformer import PedTRTransformer



def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)



# receive output from transformer 
# return the detected coord and class results to pedtr 

class PedTRHead(nn.Module): 
    def __init__(self, embed_dims=512, out_dims=2, out_dims_class=3, num_decoder_layer=6):
        super(PedTRHead, self).__init__() 

        self.transformer = PedTRTransformer()
        
        self.coord_regressor = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), 
            nn.ReLU(), 
            nn.Linear(embed_dims, embed_dims), 
            nn.ReLU(), 
            nn.Linear(embed_dims, out_dims),
        )

        self.reg_branches = nn.ModuleList([self.coord_regressor for i in range(num_decoder_layer)]) 
        self.cls_branches = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, out_dims_class),
        )


    def forward(self, img_features, proj_mats, query, query_pos):         
        
        inter_query, init_reference_point, inter_references_point = self.transformer(img_features=img_features, proj_mat=proj_mats, query=query, query_pos=query_pos, reg_branches=self.reg_branches)  
        #print(inter_query.shape, init_reference_point.shape, inter_references_point.shape) # torch.Size([100, 512]) torch.Size([100, 2]) torch.Size([100, 2])
         
        #inter_query = inter_query.permute(1, 0, 2)
        # use the output from inter query for coordinate and class regression 
        #reference = inter_references_point 
        #reference = inverse_sigmoid(inter_references_point)
        #temp = self.reg_branches[-1](inter_query) 
        #temp = temp + reference 
        #temp = temp.sigmoid()
        
        outputs_class = self.cls_branches(inter_query).unsqueeze(dim=0) # torch.Size([1, 100, 3])
        outputs_coords = inter_references_point.unsqueeze(dim=0)# torch.Size([1, 100, 2])
        
         
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coords}
        return out
    
def test(): 
    #reference_points = torch.rand((100, 2))
    query = torch.rand((100, 512))
    query_pos = torch.rand((100, 512))
    proj_mat = torch.rand((1, 7, 3, 3))
    img_feats = torch.rand((7, 512, 135, 240))
    model = PedTRHead() 
    print(model(img_feats, proj_mat, query, query_pos))
#test()