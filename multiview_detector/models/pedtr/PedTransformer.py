import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from multiview_detector.models.resnet import *
from multiview_detector.loss.pedtr.matcher import * 
from multiview_detector.loss.pedtr.criterion import SetCriterion



def init_weight(m):
        """Default initialization for Parameters of Module."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)



def ray_encoded_img(img, ray):
    assert len(img.shape) == 5 
    ray_encoded_img = []

    # Iterate over each batch
    for i in range(img.size(0)):
        # Initialize an empty list to store the concatenated tensors for the current batch
        concatenated_tensors = []
        
        # Iterate over each element along the first dimension
        for j in range(img.size(1)):
            # Concatenate tensor1[i, j] and tensor2[i, j]
            concatenated = torch.cat((img[i, j], ray[i, j]), dim=0)
            concatenated_tensors.append(concatenated)
        
        # Convert the list of tensors into a single tensor for the current batch
        combined_batch = torch.stack(concatenated_tensors, dim=0)
        ray_encoded_img.append(combined_batch)

    # Convert the list of batch tensors into a single tensor
    ray_encoded_img = torch.stack(ray_encoded_img, dim=0)
    return ray_encoded_img




class Query_genrator(nn.Module):
    def __init__(self, img_backbone=None, num_query=None, channels=None, category="naive"):
        super(Query_genrator, self).__init__()
        self.img_backbone = img_backbone
        self.category = category 
        self.embedding = nn.Parameter(torch.randn(num_query, channels))
        '''
        self.multi_head_attention=None
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(),
            nn.Linear(),
            nn.ReLU(),
            nn.LayerNorm(),
            nn.Linear()
        )
        self.mlp2 = nn.Sequential(                
            nn.LayerNorm(),
            nn.Linear(),
            nn.ReLU(),
            nn.LayerNorm(),
            nn.Linear())
        '''
    def forward(self, img=None, ray=None): 
        # img  [B, N, C, H, W]
        # ray  [B, N, 6, H, W]
        if self.category == "naive": 
            return self.embedding 
        else: 
            '''
            encoded_img = ray_encoded_img(img, ray)
            encoded_feature_maps = self.backbone(encoded_img) # B, N, C+6, H, W 
            encoded_feature_map = self.multi_head_attention(encoded_feature_maps) # B X 1 X C X H X W
            C, H, W = encoded_feature_map.shape[2], encoded_feature_map.shape[3], encoded_feature_map.shape[4]
            encoded_feature_vector = encoded_feature_vector.view(C, H*W)
            encoded_feature_vector = self.mlp1(encoded_feature_vector) # C X M

            encoded_feature_vector = encoded_feature_vector.view(-1, C) # M X C 
            # insert latent embedding 
            encoded_feature_vector = torch.cat(encoded_feature_vector, self.embedding, dim=-1) # M X 2C
            query = self.mlp2(encoded_feature_vector)  # M X C
            return query
            '''
            return self.embedding





class PreNorm(nn.Module):
    def __init__(self, dim, fn): 
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs): 
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout), 
        )
    def forward(self, x):
        return self.mlp(x) 

class PedTransformer(nn.Module): 
    def __init__(self, args, num_points=1, grid_shape=[480, 1440], num_heads=4):
        super(PedTransformer, self).__init__() 

        self.grid_shape = grid_shape
        self.num_cams = args.num_cams
        self.img_reduced_size = args.img_reduce
        self.num_query = args.num_queries
        self.num_points = num_points
        self.num_heads = num_heads
        self.dropout=args.dropout
        self.embed_dims = args.embed_dims
        self.cls_out_channels = 2
        self.reg_out_channels = 2
        self.org_img_res = [1080, 1920]
        self.device = args.device
       
        self.query_gen = Query_genrator(num_query=self.num_query, channels=self.embed_dims)
        
        self.img_backbone  = nn.Sequential(*list(resnet18(pretrained=True,
                                                     replace_stride_with_dilation=[False, True, True]).children())[:-2])
        self.MLPs_query_to_ground_coordinates =  PreNorm(self.embed_dims, FeedForward(self.embed_dims, hidden_dim=128, output_dim=2))
        self.deformable_transformer = None 
        self.query = self.query_gen()

        self.multi_head_attn = nn.MultiheadAttention(embed_dim=self.embed_dims, num_heads=4, dropout=self.dropout, batch_first=True)
        self.dropout=nn.Dropout(self.dropout)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.sigmoid = nn.Sigmoid()
        # output branches: classification and regression
        
        self.cls_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=False),
            nn.Linear(self.embed_dims, self.cls_out_channels),
        ) 
        '''
        self.cls_branch = nn.Linear(self.embed_dims, self.cls_out_channels)
        self.reg_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=False),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=False),
            nn.Linear(self.embed_dims, self.reg_out_channels)
        )

        '''
        self.reg_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=False),
            nn.Linear(self.embed_dims, self.reg_out_channels),
        ) 
        
         
    def query_generator(self, ):
        pass 
    # img: B X N X C X H X W  multiview images 
    # ray: B X N X 6 X H X W  camera ray for per pixel per image, including (x,y,z), (ex,ey,ez)
    # proj_mat: N X 3 X 3, project ground grid into image coordinates 
    def feature_sampling(self, ground_coordinates, proj_mat, img_feats):

        assert len(proj_mat.shape) == 4 and proj_mat.shape[1] == self.num_cams

        ground_coordinates = torch.cat((ground_coordinates, torch.ones_like(ground_coordinates[..., :1])), dim=1)
        img_coord =  proj_mat[0].float() @ ground_coordinates.T # shape torch.Size([7, 3, 100])
        img_coord = torch.transpose(img_coord, 1, 2)
        
        reference_points_ground = ground_coordinates.clone()
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
        
        mask = mask.view(1, self.num_cams, 1, self.num_query, 1).permute(0, 2, 3, 1, 4)
        mask = torch.nan_to_num(mask)

        # feature sampling 
        BN, C, H, W = img_feats.size()
        reference_points_cam_lvl = reference_points_cam.view(BN, self.num_query, 1, 2)
        sampled_feat = F.grid_sample(img_feats, reference_points_cam_lvl, align_corners=False)
        sampled_feat = sampled_feat.view(1, C, self.num_query, self.num_cams, 1) # 1, 512, 200, N, 1
         
        return reference_points_ground, sampled_feat, mask
    
    def forward(self, img, ray=None, proj_mat=None):
        with torch.autograd.set_detect_anomaly(True):
            B, N, C, H, W = img.shape
            img = img.reshape(B*N, C, H, W)
            
            # image feature extraction 
            img_features =  self.img_backbone(img)
            
            # queries generation 
            queries = self.query
            # ground grid coordiantes generation
            ground_coordinates = self.sigmoid(self.MLPs_query_to_ground_coordinates(queries))
            ground_coordinates = ground_coordinates.clone() # without this line, causing inplace error 
            ground_coordinates[:, 0] =  ground_coordinates[:, 0] * self.grid_shape[0]
            ground_coordinates[:, 1] =  ground_coordinates[:, 1] * self.grid_shape[1]
            ground_coordinates = ground_coordinates.to(self.device)
            
            
            reference_points_ground, output, mask = self.feature_sampling(ground_coordinates, proj_mat, img_features)
            output = torch.nan_to_num(output) # torch.Size([1, 512, 200, 7, 1])
            #mask = torch.nan_to_num(mask) # torch.Size([1, 1, 200, 7, 1])
            
            mask = ~mask
            
            output = output.view(self.num_query, self.num_cams, -1)
            mask = mask.view(self.num_query, 1, self.num_cams)
            mask = mask.repeat(self.num_heads, 1, 1).repeat(1, self.num_cams, 1) 
            #print(output.shape, mask.shape)  #torch.Size([100, 7, 512]) torch.Size([400, 7, 7])
            output, output_weights = self.multi_head_attn(query=output, key=output, value=output, attn_mask=mask) # torch.Size([200, 7, 512]) torch.Size([200, 7, 7])
           
            output = torch.mean(output, dim=1, keepdim=True)
            output = output.permute(1, 0, 2) # torch.Size([1, 200, 512])
            output = self.dropout(self.output_proj(output)) + queries.unsqueeze(dim=0)    # torch.Size([1, 200, 512])
            
           
            #print(output.shape) # torch.Size([1, 200, 512])
            #retrieve image features based on perspective projection

            #deformable attention and cross attention w.r.t the perspective projection and ground_coordinates 
            #update query 
            outputs_class = self.cls_branch(output)
            outputs_coords = self.reg_branch(output).sigmoid()
            #print(outputs_class.shape,outputs_coords.shape)
            #exit()
            #print(outputs_class.shape, outputs_coords.shape)
            
            out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coords}
            return out
        


def build_model(args): 

    device = torch.device(args.device)
    # build_model  
    model = PedTransformer(args).to(device)
    # build matcher 
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    #losses = ['labels', 'boxes']
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes=1, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    return model, criterion

   



'''

def test(): 
    from multiview_detector.datasets_pedtr.ped_dataset import PedestrianDataset
    from torch.utils.data import DataLoader 
    from torchvision import transforms as T
    root = '../Data/Wildtrack'
    num_cam = 7
    num_frame = 2000 
    img_shape = [1080, 1920] 
    world_grid_shape = [480, 1440] 
    img_reduce = 4
    grid_reduce =1 
    transform = T.Compose([
        T.ToTensor(),]
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),]
    )
    train_ratio = 0.9 
    reID = False
    intrinsic_camera_matrix_filenames =['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
    extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']
    is_train = True #False
    force_download=False
    dataset = PedestrianDataset(root, num_cam, num_frame, img_shape, world_grid_shape, img_reduce, 
                 grid_reduce, transform, train_ratio, reID, intrinsic_camera_matrix_filenames, 
                 extrinsic_camera_matrix_filenames, is_train)

    print(dataset)
   

    ped_model = PedTransformer()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (imgs, proj_mats, frame, map_gt) in enumerate(dataloader):
        print(imgs[i].shape, proj_mats[0]. shape)

        out = ped_model(img=imgs, proj_mat=proj_mats)
        print(out)
        break


    #ped_model = MultiheadAttention(embed_dim=512, num_heads=4, dropout=0, batch_first=True)
    #ped_model = PedTransformer()
    #inputs = torch.rand(200, 7, 512)
    #out, attn_score = ped_model(inputs, inputs, inputs)
    #print(out.shape)
    #out_fusion = torch.mean(out, dim=1)
    #print(out_fusion.shape)
'''
    

#test()