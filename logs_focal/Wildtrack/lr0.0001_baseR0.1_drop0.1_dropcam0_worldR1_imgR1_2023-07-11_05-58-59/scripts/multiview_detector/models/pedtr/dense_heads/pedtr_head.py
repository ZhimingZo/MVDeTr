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
    def __init__(self, args):
        super(PedTRHead, self).__init__() 

        self.embed_dims = args.embed_dims  # 512 
        self.out_dims_coord =  2
        self.out_dims_class =  2
        self.num_decoder_layer=args.num_decoder_layer # 6 
 
        self.transformer = PedTRTransformer(args)
        
        self.coord_regressor = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.ReLU(), 
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.ReLU(), 
            nn.Linear(self.embed_dims, self.out_dims_coord),
        )
        self.cls_regressor = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.out_dims_class),
        )

        self.reg_branches = nn.ModuleList([self.coord_regressor for i in range(self.num_decoder_layer)]) 
        self.cls_branches = nn.ModuleList([self.cls_regressor  for i in range(self.num_decoder_layer)]) 

    def forward(self, img_features, proj_mats, query, query_pos):         
        
        inter_queries, init_reference_point, inter_references_points = \
            self.transformer(img_features=img_features, proj_mat=proj_mats, query=query, query_pos=query_pos, reg_branches=self.reg_branches)  
        # inter_queries, inter_references_points # torch.Size([6, 100, 512]) torch.Size([6, 100, 2])

        out = []
        for lvl in range(inter_queries.shape[0]): 
            if lvl == 0: 
                reference = init_reference_point 
            else: 
                reference = inter_references_points[lvl-1]

            reference = inverse_sigmoid(reference)
            output_class = self.cls_branches[lvl](inter_queries[lvl])
            tmp = self.reg_branches[lvl](inter_queries[lvl])

            assert reference.shape[-1] == 2 
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp = tmp.sigmoid() 
            output_coord = tmp

            output_class = output_class.unsqueeze(0) 
            output_coord = output_coord.unsqueeze(0) 
            
            out_lvl = {'pred_logits':  output_class, 'pred_boxes': output_coord}
            out.append(out_lvl)
        return out
    
def test(): 
    query = torch.rand((100, 512))
    query_pos = torch.rand((100, 512))
    proj_mat = torch.rand((1, 7, 3, 3))
    img_feats = torch.rand((7, 512, 135, 240))
    model = PedTRHead() 
    print(model(img_feats, proj_mat, query, query_pos))
