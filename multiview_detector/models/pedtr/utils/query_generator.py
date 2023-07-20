import torch 
import torch.nn as nn 
from multiview_detector.models.pedtr.backbones.resnet import resnet18
import torch.nn.functional as F 
def ray_encoded_img(img, ray):
    assert len(img.shape) == 4 
    ray_encoded_img = []
   
    concatenated_tensors = []
    # Iterate over each element along the first dimension
    for j in range(img.size(0)):
        # Concatenate tensor1[i, j] and tensor2[i, j]
        concatenated = torch.cat((img[j], ray[j]), dim=0)
        concatenated_tensors.append(concatenated)
        
    # Convert the list of batch tensors into a single tensor
    ray_encoded_img = torch.stack(concatenated_tensors,  dim=0)
    return ray_encoded_img



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

class FeatureTransformerLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super(FeatureTransformerLayer, self).__init__() 
        self.layers = nn.ModuleList([])
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

class FeatureTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super(FeatureTransformer, self).__init__() 
        self.layers = nn.ModuleList([FeatureTransformerLayer(dim, heads, mlp_dim, dropout) for i in range(depth)])
    def forward(self, x, mask=None): 
        output = x
        for layer in self.layers: 
            output = layer(x=output, mask=mask)
        return output
    
class FeatureCrossTransformerLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super(FeatureCrossTransformerLayer, self).__init__() 
        #self.embedding = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
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
    def forward(self, input1, input2, mask=None):
        org_x = input1
        input2 = self.layer_norm(input2)
        for norm1, attn, norm2, ff in self.layers:
            x = norm1(input1)
            x, _ = attn(query=x, key=input2, value=input2, attn_mask=mask) 
            x = x + org_x
            x = ff(norm2(x)) + x
            org_x = x
        return x 

class FeatureCrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super(FeatureCrossTransformer, self).__init__() 
        self.layers = nn.ModuleList([FeatureCrossTransformerLayer(dim, heads, mlp_dim, dropout) for i in range(depth)])
    def forward(self, sq1, sq2, mask=None): 
        output = sq1 
        for layer in self.layers: 
            output = layer(input1=output, input2=sq2, mask=mask) + output
        return output



class Query_generator(nn.Module):
    def __init__(self, args, num_query=None, dims=None):
        super(Query_generator, self).__init__()
        
        self.args = args 
        #self.img_backbone = img_backbone[1:]
        #self.conv1 = nn.Conv2d(6, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(2,2)) 
        self.embed_dims = dims
         
        self.output_proj = nn.Sequential(            
            nn.Linear(self.embed_dims+3, self.embed_dims*2), 
            nn.ReLU(), 
            nn.Dropout(p=self.args.dropout), 
            nn.Linear(self.embed_dims*2, self.embed_dims), 
            nn.Dropout(p=self.args.dropout), 
        )

        self.attn = FeatureTransformer(dim=self.embed_dims , depth=2, heads=4, mlp_dim=self.embed_dims, dropout=self.args.dropout)
        self.cross_attn = FeatureCrossTransformer(dim=self.embed_dims , depth=2, heads=4, mlp_dim=self.embed_dims, dropout=self.args.dropout)
        self.ffns_query = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.ReLU(), 
            nn.Dropout(p=self.args.dropout), 
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.Dropout(p=self.args.dropout), 
        )
        self.dropout = nn.Dropout(self.args.dropout)
        
        self.query_embedding = nn.Embedding(num_query, self.embed_dims)
        self.pos_embedding = nn.Embedding(num_query, self.embed_dims)
       

    def forward(self, img_feat=None, ray=None): 
        # img  [B, N, C, H, W]
        # ray  [B, N, 6, H, W]
        if ray == None: 
            return self.query_embedding, self.pos_embedding
        else:
            #pooled_encoded_feature_maps = self.maxPool(img_feat) # N X C X 67 X 120 
            query = self.query_embedding.weight
            query_pos = self.pos_embedding.weight
            B, N, C, H, W = ray.shape
            ray = ray.view(B*N,C, H, W)
            encoded_img = ray_encoded_img(img_feat, ray)
            pooled_encoded_feature_maps = self.avg_pool2d(encoded_img)
            BN, C1, H1, W1 = pooled_encoded_feature_maps.shape
            pooled_encoded_feature_maps = pooled_encoded_feature_maps.view(BN, C1, -1).permute(0, 2, 1)
            pooled_encoded_feature_maps = self.output_proj(pooled_encoded_feature_maps)

            pooled_encoded_feature_maps = pooled_encoded_feature_maps.permute(1, 0, 2) # 8040 x7 x 512

            pooled_encoded_feature_maps = self.dropout(self.attn(pooled_encoded_feature_maps)) + pooled_encoded_feature_maps
            pooled_encoded_feature_maps = torch.mean(pooled_encoded_feature_maps, dim=1)
            query_embeddings = self.cross_attn(sq1=query, sq2=pooled_encoded_feature_maps) 
            query_embeddings = self.dropout(self.ffns_query(query_embeddings)) + query_embeddings

            return query_embeddings, query_pos


#print(list(Query_generator(num_query=100, dims=512).parameters()))
def test(): 
    tensor = torch.randn(1,7, 3,1080, 1920)
    ray = torch.randn(1, 7, 6, 1080, 1920)
    backbone = nn.Sequential(*list(resnet18(pretrained=True, replace_stride_with_dilation=[False, True, True]).children())[:-2])
    Query =  Query_generator(img_backbone=backbone, num_query=100, dims=512, category='ray')

    query, query_pos = Query(tensor, ray)
    print(query.shape)


    pass 
#test()
