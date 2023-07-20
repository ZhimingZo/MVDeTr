import math
import torch
from torch import nn

def get_rays_new(image_size, H, W, K, R, T, ret_rays_o=False):
    # calculate the camera origin
    ratio = W / image_size[1]
    batch = K.size(0)
    views = K.size(1)
    K = K.reshape(-1, 3, 3).float()
    R = R.reshape(-1, 3, 3).float()
    T = T.reshape(-1, 3, 1).float()
    # re-scale camera parameters
    K[:, :2] *= ratio
    rays_o = -torch.bmm(R.transpose(2, 1), T)
    # calculate the world coordinates of pixels
    j, i = torch.meshgrid(torch.linspace(0, H-1, H),
                          torch.linspace(0, W-1, W))
    xy1 = torch.stack([i.to(K.device), j.to(K.device),
                       torch.ones_like(i).to(K.device)], dim=-1).unsqueeze(0)
    
    pixel_camera = torch.bmm(xy1.flatten(1, 2).repeat(views, 1, 1),
                             torch.inverse(K).transpose(2, 1))
    pixel_world = torch.bmm(pixel_camera-T.transpose(2, 1), R)
    rays_d = pixel_world - rays_o.transpose(2, 1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = rays_o.unsqueeze(1).repeat(1, H*W, 1, 1)
    if ret_rays_o:
        return rays_d.reshape(batch, views, 3, H, W), \
               rays_o.reshape(batch, views, 3, H, W) / 1000
    else:
        return rays_d.reshape(batch, views, 3, H, W)
    
def test():
    img_size = [1080, 1920]
    H = 1080 
    W = 1920 
    K = torch.randn(1,7,3,3)
    R = torch.randn(1,7,3,3)
    T = torch.randn(1,7,3,1)

    d, o = get_rays_new(img_size, H, W, K, R, T, ret_rays_o=True)
    print(d.shape, o.shape)

#test()