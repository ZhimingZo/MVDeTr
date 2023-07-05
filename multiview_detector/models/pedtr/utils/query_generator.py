import torch 
import torch.nn as nn 

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


class Query_generator(nn.Module):
    def __init__(self, img_backbone=None, num_query=None, dims=None, category="naive"):
        super(Query_generator, self).__init__()
        self.img_backbone = img_backbone
        self.category = category 
        self.query_embedding = nn.Parameter(torch.randn(num_query, dims))
        self.pos_embedding = nn.Parameter(torch.randn(num_query, dims))
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
            return self.query_embedding, self.pos_embedding
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
            return self.query_embedding, self.pos_embedding



#print(list(Query_generator(num_query=100, dims=512).parameters()))