import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
# Define camera intrinsics
focal_length_x = 500.0
focal_length_y = 500.0
principal_point_x = 320.0
principal_point_y = 240.0
intrinsic_matrix = torch.tensor([[focal_length_x, 0, principal_point_x],
                                 [0, focal_length_y, principal_point_y],
                                 [0, 0, 1]])

# Define camera extrinsics
camera_position = torch.tensor([0.0, 0.0, 0.0])
R = torch.eye(3)  # Identity rotation matrix
t = torch.tensor([0.0, 0.0, 1.0])  # Translation vector
extrinsic_matrix = torch.cat([R, t.unsqueeze(1)], dim=1)
extrinsic_matrix = torch.cat([extrinsic_matrix, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)

#print(extrinsic_matrix.shape)
# Define image dimensions
image_height = 480
image_width = 640

# Generate pixel coordinates
y, x = torch.meshgrid(torch.arange(image_height), torch.arange(image_width))
x = (2 * (x + 0.5) / image_width - 1)
y = (2 * (y + 0.5) / image_height - 1)

# Compute ray directions
ray_directions = torch.stack([x, y, torch.ones_like(x)], dim=-1)

ray_directions = ray_directions @ intrinsic_matrix.inverse().t()
ray_directions = torch.cat([ray_directions, torch.ones((480, 640, 1))], dim=-1)
ray_directions = ray_directions @ extrinsic_matrix.inverse().t()
ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)

# Compute ray origins
ray_origins = torch.broadcast_to(camera_position, ray_directions.shape[:-1] + (3,))

'''


# compute camera rays 
def get_rays_new(image_size, H, W, K, R, T, ret_rays_o=False):
    # calculate the camera origin
    ratio = W / image_size[0]
    
    batch = 1#K.size(0)
     
    views = 1#K.size(1)
     
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
        return rays_d.reshape(batch, views, H, W, 3), \
               rays_o.reshape(batch, views, H, W, 3) / 1000
    else:
        return rays_d.reshape(batch, views, H, W, 3)


img_size = [1920, 1080]
H=1080
W=1920
K = torch.tensor([[1.74344788e+03, 0.00000000e+00, 9.34520203e+02],
                  [0.00000000e+00, 1.73515662e+03, 4.44398773e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
R = torch.tensor([[8.77676189e-01,  4.78589416e-01,  2.52333060e-02],
                  [ 1.33892894e-01, -1.94308296e-01, -9.71759737e-01],
                  [-4.60170895e-01,  8.56268942e-01, -2.34619483e-01  ]])
T = torch.tensor([[-5.25894165e+02], [4.54076347e+01], [9.86723511e+02]])

ray_directions, ray_origins = get_rays_new(image_size=img_size, H=H, W=W, K=K, R=R, T=T, ret_rays_o=True)




# Set up the visualization environment
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Plot the camera rays
#print(ray_directions)
print(ray_origins.shape)
print(ray_directions.shape)
#exit()
#print(ray_origins)

ax.quiver(ray_origins[..., 0], ray_origins[..., 1], ray_origins[..., 2],
          ray_directions[..., 0], ray_directions[...,1 ], ray_directions[..., 2], length=10)

 

#ax.quiver(ray_origins[:, 0, 0, 0], ray_origins[:, 0, 0, 1], ray_origins[:, 0, 0, 2],
#          ray_directions[:, 0, 0, 0], ray_directions[:, 0, 0, 1], ray_directions[:, 0, 0, 2], length=10)


#ax.quiver(ray_origins[100, 100, 0], ray_origins[100, 100, 1], ray_origins[100, 100, 2],
#          1, 2, 3, length=10)

#ax.quiver(ray_origins[100, 100, 0], ray_origins[100, 100, 1], ray_origins[100, 100, 2],
#          0.5, 0.5, 0.5, length=10)

# Set plot limits and labels
ax.set_xlim([-1, 1])  # Define the appropriate limits
ax.set_ylim([-1, 1])  # Define the appropriate limits
ax.set_zlim([-1, 1])  # Define the appropriate limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.savefig("test1.png")
# Show the plot
#plt.show()


