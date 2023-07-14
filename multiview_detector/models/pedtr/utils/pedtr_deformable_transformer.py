import torch 
import torch.nn as nn 
import torch.nn.functional as F

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

# PedTRTransformer consists of a decoder layer  
# receive updated query from  decoder  
# output: coord and cls results to pedtr_head 


class PedTRDeformTransformer(nn.Module): 
    def __init__(self, args):
        super(PedTRDeformTransformer, self).__init__() 
        self.embed_dims = args.embed_dims
        self.decoder = PedTRDeformTransformerDecoder(args)
        self.referece_points_init  = nn.Linear(self.embed_dims, 2)
    
        nn.init.xavier_uniform_(self.referece_points_init.weight)
        nn.init.zeros_(self.referece_points_init.bias)

    def forward(self, img_features, proj_mat, query, query_pos, reg_branches): 

        reference_points_init = self.referece_points_init(query).sigmoid()

        # visulizaton here 
        '''
        import numpy as np
        import cv2 
        
        reference_points_init_draw = reference_points_init.cpu().numpy()
        reference_points_init_draw[:, 0] *= 480 
        reference_points_init_draw[:, 1] *= 1440 
        print(reference_points_init_draw.shape)
        map_gt = np.zeros((480, 1440, 3), dtype=np.uint8)
        for point in reference_points_init_draw:
            #print(point[0])
            cv2.circle(map_gt, (int(point[1]), int(point[0])), 15, (0, 0, 255), -1)
        #cv2.imshow("reference_init", map_gt)
        cv2.imwrite("ce_100_query_train_epoch_best_init_ref.png", map_gt)
        #exit()
        '''
        inter_queries, inter_references_points = self.decoder(img_feats=img_features, proj_mat=proj_mat, query=query, query_pos=query_pos, \
                                                              reference_points=reference_points_init, reg_branches=reg_branches)  

        # viz for output coord
        '''
        i = 0 
        for inter_references_point in inter_references_points:
            reference_points_inter_draw = inter_references_point.cpu().numpy()
            reference_points_inter_draw[:, 0] *= 480 
            reference_points_inter_draw[:, 1] *= 1440 
            print(reference_points_inter_draw.shape)
            map_gt = np.zeros((480, 1440, 3), dtype=np.uint8)
            for point in reference_points_inter_draw:
                #print(point[0])
                cv2.circle(map_gt, (int(point[1]), int(point[0])), 15, (255, 255, 255), -1)
            #cv2.imshow("reference_init", map_gt)
            cv2.imwrite("ce_100_query_epoch_best_inter_ref_"+str(i)+".png", map_gt)
            i = i+1 
        exit(0)
        '''

        return  inter_queries, reference_points_init, inter_references_points
    

# PedTRTransformerDecoder consists of n layers decoder layer 

# each layer will do following 

# step1: estimate 3D (the 3rd dimension z=0 can be discard) ground referece points 
# step2: project 3D points to 2D images using camera transformation matrices 
# step3: sampling features from multiple view image w.r.t the estimated projected 2D image points  
# step4: update sampled image features with transformer(multi-head attention + ffns)
# step5: fuse the updated image features (avg pooling)  
# step6: update query 

class PedTRDeformTransformerDecoder(nn.Module): 
    def __init__(self, args=None, return_intermediate=True):
        super(PedTRDeformTransformerDecoder, self).__init__() 
        self.return_intermediate = return_intermediate
        self.num_decoder_layer = args.num_decoder_layer # 6 
        self.decoder_layer = PedTRDeformTransformerDecoderLayer(args) 
        self.layers = nn.ModuleList(self.decoder_layer for i in range(self.num_decoder_layer))  
        

    def forward(self, img_feats, proj_mat, query, query_pos, reference_points, reg_branches=None):    
        output = query 
        intermediate = []
        intermediate_reference_points = []
        for idx, decoder_layer in enumerate(self.layers): 
            reference_points_input = reference_points 
            output = decoder_layer(img_feats=img_feats, proj_mat=proj_mat, query=output, query_pos=query_pos, reference_points=reference_points_input) 
            if reg_branches is not None:
                tmp = reg_branches[idx](output)  
                
                assert reference_points.shape[-1] == 2
               
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
    
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate: 
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        return output, reference_points
    

class PedTRDeformTransformerDecoderLayer(nn.Module):
    def __init__(self, args=None, num_points=4): 
        super(PedTRDeformTransformerDecoderLayer, self).__init__()
        
        self.embed_dims = args.embed_dims # 512 
        self.dropout_ratio = args.dropout # 0.1 
        self.num_heads = args.num_heads # 4 
        self.num_cams = args.num_cams #7 
        self.num_query = args.num_queries # 100
        self.org_img_res = args.org_img_shape # 1080 1920 
        self.grid_shape = args.world_grid_shape #480 1440 
        self.num_points = num_points
        '''
        self.embed_dims = 512 
        self.dropout_ratio = 0.1 
        self.num_heads = 4 
        self.num_cams = 7 
        self.num_query = 100
        self.org_img_res = [1080, 1920] 
        self.grid_shape = [480, 1440] 
        self.num_points = num_points
        '''
        # MultiHeadAttn 
        self.multiheadattn_query = nn.MultiheadAttention(self.embed_dims, num_heads=self.num_heads)
        # deformable feature fusion 
        self.deformable_reference_offset = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.ReLU(), 
            nn.Dropout(p=self.dropout_ratio), 
            nn.Linear(self.embed_dims, self.num_points*2) 
        )
        self.deformable_features_fusion_attention_mask = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.ReLU(), 
            nn.Dropout(p=self.dropout_ratio), 
            nn.Linear(self.embed_dims, self.num_points) 
        )

        # image feature sampling 
        self.img_feature_transformer = ImgFeatureTransformer(dim=self.embed_dims, depth=4, heads=self.num_heads, mlp_dim=self.embed_dims, dropout=self.dropout_ratio)

        # feedforward
        self.ffns_query = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.ReLU(), 
            nn.Dropout(p=self.dropout_ratio), 
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.Dropout(p=self.dropout_ratio), 
        )
        
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        # postional encoder 
        self.position_encoder = nn.Sequential(
            nn.Linear(2, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(), 
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims), 
            nn.ReLU() 
        )

        # Layer norm 
        self.layerNorm1 = nn.LayerNorm(self.embed_dims)
        self.layerNorm2 = nn.LayerNorm(self.embed_dims)
        self.layerNorm3 = nn.LayerNorm(self.embed_dims)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    def forward(self, img_feats, proj_mat, query, query_pos, reference_points):
        #inp_residual = query 
        if query_pos is not None: 
            query = query + query_pos  # torch.Size([100, 512])
            #print(query.shape, query_pos.shape)
        # query multi-head attention 
         
        query_out, _ = self.multiheadattn_query(query, query, query)
        query = self.dropout(query_out) + query
        query = self.layerNorm1(query) # print(query.shape)    
        
        #generate deformable reference points offset 
        offset = self.deformable_reference_offset(query) # 100 x 8   
        offset = offset.view(-1, 2) # 400 x 2
        reference_points_deform = inverse_sigmoid(reference_points).repeat(4, 1) + offset #(400, 2)
        reference_points_deform = reference_points_deform.sigmoid()
        attn_mask = self.deformable_features_fusion_attention_mask(query) # num_query x 4
        attn_mask = F.softmax(attn_mask, dim=-1)
        '''
        print("query:", query.shape) #torch.Size([100, 512])
        print("offset: ", offset.shape) # torch.Size([400, 2])
        print("reference_points_deform: ", reference_points_deform.shape) #  torch.Size([400, 2])
        print("attn_mask: ", attn_mask.shape) # torch.Size([100, 4])
        '''
        # update query and extracted img feature attention 
        reference_points_3d, output, mask = self.feature_sampling(reference_points_deform, proj_mat, img_feats)
        output = torch.nan_to_num(output)
        mask = ~mask
        #print("ref_3d: ", reference_points_3d.shape)  # torch.Size([400, 2])
        #print("output: ", output.shape) # torch.Size([400, 7, 512, 1])
        #print("mask: ", mask.shape) #  mask:   torch.Size([4, 100, 1, 7])
      
        output = output.view(self.num_points * self.num_query, self.num_cams, -1) # 400 x 7 x 512
        mask = mask.repeat(1, self.num_heads, 1, 1).repeat(1, 1, self.num_cams, 1)  # 400, 7, 7
        mask = mask.view(-1, self.num_cams, self.num_cams)
        #print("output: ", output.shape) # torch.Size([400, 7, 512])
        #print("mask: ", mask.shape) #  mask:  torch.Size([1600, 7, 7])

        output = self.img_feature_transformer(x=output, mask=mask) # torch.Size([400, 7, 512])
        output = torch.mean(output, dim=1, keepdim=False) # torch.Size([400, 512])
        output = output.view(self.num_query, self.num_points, -1) # 100 X 4 x 512 
        attn_mask = attn_mask.view(self.num_query, self.num_points, 1)
        output = output * attn_mask 
        output = torch.mean(output, dim=1, keepdim=False)
        
        #output = self.img_feature_transformer(x=output, mask=mask) # torch.Size([100, 7, 512])
        #output = torch.mean(output, dim=1, keepdim=False) # torch.Size([100, 1, 512])
        #output = torch.permute(output, (1, 0, 2)) # torch.Size([1, 100, 512])
        output = self.output_proj(output) # torch.Size([1, 100, 512])
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d))
        query = self.dropout(output) + query + pos_feat 
        query = self.layerNorm2(query)

        # ffn projection 
        query = self.dropout(self.ffns_query(query)) + query
        query = self.layerNorm3(query)
        #print(query.shape)
         
        return query
    
    def feature_sampling(self, ground_coordinates, proj_mat, img_feats):

        assert len(proj_mat.shape) == 4 and proj_mat.shape[1] == self.num_cams
        reference_points_ground = ground_coordinates.clone()
        reference_points = ground_coordinates.clone() 
        reference_points[..., 0] = reference_points[..., 0] * self.grid_shape[0] #480
        reference_points[..., 1] = reference_points[..., 1] * self.grid_shape[1] #1440
        reference_points  = torch.cat((reference_points , torch.ones_like(reference_points [..., :1])), dim=-1)
        
        img_coord =  proj_mat[0].float() @ reference_points.T # shape torch.Size([7, 3, 100])
        img_coord = torch.transpose(img_coord, 1, 2) #  shape torch.Size([7, 100, 3])
        
        reference_points_cam = img_coord 

        eps = 1e-5
        mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps) #  [x, y] in image space 

        reference_points_cam[..., 0] = reference_points_cam[..., 0] / self.org_img_res[1] 
        reference_points_cam[..., 1] = reference_points_cam[..., 1] / self.org_img_res[0]
        reference_points_cam = (reference_points_cam - 0.5) * 2
        mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))

        mask = mask.view(self.num_cams, 1, self.num_query, self.num_points).permute(3, 2, 1, 0)
        mask = torch.nan_to_num(mask)
        # feature sampling 
        BN, C, H, W = img_feats.size()
        reference_points_cam_lvl = reference_points_cam.view(BN, self.num_query*self.num_points, 1, 2)
        sampled_feat = F.grid_sample(img_feats, reference_points_cam_lvl, align_corners=False) # torch.Size([7, 512, 400, 1])
        sampled_feat = sampled_feat.permute(2, 0, 1, 3)
        #sampled_feat = sampled_feat.view(1, self.num_cams, C, self.num_query) # 1, 512, 200, N, 1
        return reference_points_ground, sampled_feat, mask
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout), 
        )
        self.reset_parameters()
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    def forward(self, x):
        return self.mlp(x) 
    
class ImgFeatureTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super(ImgFeatureTransformer, self).__init__() 
        self.layers = nn.ModuleList([])
        for _  in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True), 
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dim, dropout=dropout),
            ]))
        self.reset_parameters()
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    def forward(self, x, mask=None):
        org_x = x 
        for norm1, attn, norm2, ff in self.layers:
            x = norm1(x)
            x, _ = attn(query=x, key=x, value=x, attn_mask=mask) 
            x = x + org_x
            x = ff(norm2(x)) + x
            org_x = x
        return x 


def test():
    
    
    # test extracted img feature for MultiHeadAttn 
    
    '''
    imgs = torch.rand((100, 7, 512))
    mask = torch.rand((400, 7, 7))
    ImgFeatureTransformerModel = ImgFeatureTransformer(dim=512, depth=3, heads=4, mlp_dim=512, dropout=0.1)
    print(ImgFeatureTransformerModel)
    print(ImgFeatureTransformerModel(imgs, mask=mask).shape) # [100, 512]
    '''

    # decoder layer 
    
    
    PedTRDeformTransformerDecoderLayerModel = PedTRDeformTransformerDecoderLayer()
    #print(PedTRTransformerDecoderLayerModel)
    reference_points = torch.rand((100, 2))
    query = torch.rand((100, 512))
    query_pos = torch.rand((100, 512))
    proj_mat = torch.rand((1, 7, 3, 3))
    img_feats = torch.rand((7, 512, 135, 240))
    out = PedTRDeformTransformerDecoderLayerModel(img_feats=img_feats, proj_mat=proj_mat, query=query, query_pos=query_pos, reference_points=reference_points)
    print(out, out.shape) #torch.Size([1, 100, 512])
    
    
    # decoder 
    '''
    num_decoder_layer = 6
    embed_dims=512
    out_dims=2 
    coord_regressor = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), 
            nn.ReLU(), 
            nn.Linear(embed_dims, embed_dims), 
            nn.ReLU(), 
            nn.Linear(embed_dims, out_dims),
        )

    reg_branches = nn.ModuleList([coord_regressor for i in range(num_decoder_layer)]) 
    reference_points = torch.rand((100, 2))
    query = torch.rand((100, 512))
    query_pos = torch.rand((100, 512))
    proj_mat = torch.rand((1, 7, 3, 3))
    img_feats = torch.rand((7, 512, 135, 240))
    PedTRTransformerDecoderModel = PedTRTransformerDecoder() 
    #print(PedTRTransformerDecoderModel())
    out, out2 = PedTRTransformerDecoderModel(reference_points=reference_points, query=query, query_pos=query_pos, proj_mat=proj_mat, img_feats=img_feats)
    #print(out.shape, out2.shape)


    PedTRTransformerModel = PedTRTransformer()
    A, B, C = PedTRTransformerModel(img_features=img_feats, proj_mat=proj_mat, query=query, query_pos=query_pos, reg_branches=reg_branches)
    print(A.shape, B.shape, C.shape)
    '''
#test()
