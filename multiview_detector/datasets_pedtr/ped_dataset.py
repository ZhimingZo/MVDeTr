from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
import os
import json
import numpy as np 
import cv2
import xml.etree.ElementTree as ET 
from PIL import Image 
import argparse
from torch.utils.data import DataLoader 




class PedestrianDataset(Dataset): 

    def __init__(self, args, root, org_img_shape, world_grid_shape,  
                transform, intrinsic_camera_matrix_filenames, 
                 extrinsic_camera_matrix_filenames, is_train): 
        super().__init__()
       
        self.intrinsic_camera_matrix_filenames = intrinsic_camera_matrix_filenames
        self.extrinsic_camera_matrix_filenames = extrinsic_camera_matrix_filenames

        self.reID, self.grid_reduce, self.img_reduce = args.reID, args.world_grid_reduce, args.img_reduce
        self.root, self.num_cam, self.num_frame = root, args.num_cams, args.num_frames
        self.train_ratio = args.train_ratio
        self.img_shape, self.world_grid_shape = org_img_shape, world_grid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.world_grid_shape))
        self.indexing = 'ij'
        self.transform =  transform
        self.is_train = is_train
        self.reID = args.reID
        self.ID = {}
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
         
        if self.is_train:
            frame_range = range(0, int(self.num_frame * self.train_ratio))
        else:
            frame_range = range(int(self.num_frame * self.train_ratio), self.num_frame)


        # prepare for images
        self.img_fpaths = self.get_image_fpaths(frame_range)
        self.map_gt={}
        self.download(frame_range)


        # prepare for cams 
        self.worldgrid2worldcoord_mat = np.array([[2.5, 0, -300], [0, 2.5, -900], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])
      


    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths
    
    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.get_world_grid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')
    
    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                for single_pedestrian in all_pedestrians:
                    x, y = self.get_world_grid_from_pos(single_pedestrian['positionID'])
                    if self.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                self.map_gt[frame] = np.column_stack([i_s, j_s])
                self.ID[frame] = v_s
                

    def __getitem__(self, index):

        filename=[]
        frame = list(self.map_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            filename.append(fpath)
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs =  torch.stack(imgs) # shape torch.Size([7, 3, 1080, 1920])
        boxes = torch.from_numpy(self.map_gt[frame]).float()
        #   normalize 
        boxes[:, 0] = boxes[:, 0] / self.world_grid_shape[0]
        boxes[:, 1] = boxes[:, 1] / self.world_grid_shape[1]
        labels = torch.ones(boxes.shape[0], dtype=torch.long)
        
        if self.reID:
            ID = self.ID[frame]
        worldgrid2imgcoord_matrices = self.get_worldgrid2imagecoord_matrices(self.intrinsic_matrices,
                                                                             self.extrinsic_matrices,
                                                                             self.worldgrid2worldcoord_mat)
        worldgrid2imgcoord_matrices = torch.stack(worldgrid2imgcoord_matrices, dim=0)

        gt = {
            'boxes': boxes,
            'labels': labels,
        } 

        return imgs, worldgrid2imgcoord_matrices, gt, frame 

    def __len__(self):
        return len(self.map_gt.keys())
    
    def get_world_grid_from_pos(self, posID):
        grid_x = posID % self.world_grid_shape[0]
        grid_y = posID // self.world_grid_shape[0]
        return np.array([grid_x, grid_y], dtype=int)
    
    def get_worldgrid2imagecoord_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        #projection_matrices = {}
        projection_matrices = []
        for cam in range(self.num_cam):
            # step 1 # convert  world grid to world coordinates (# Given augument)
            # step 2 # convert  world coordinates to image coordinates   
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            projection_matrices.append(torch.from_numpy(worldgrid2imgcoord_mat))
        return projection_matrices 
    
    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             self.intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           self.extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))
        #print(extrinsic_matrix.shape, intrinsic_matrix.shape)
        return intrinsic_matrix, extrinsic_matrix
    
def get_worldcoord_from_worldgrid(worldgrid):
    # datasets default unit: centimeter & origin: (-300,-900)
    grid_x, grid_y = worldgrid[:, 0], worldgrid[:, 1]
    coord_x = -300 + 2.5 * grid_x
    coord_y = -900 + 2.5 * grid_y
    return np.array([coord_x, coord_y])


def build_dataset(isTrain, args):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),]
    )
    if 'Wildtrack' in args.dataset:
        dataset_root=os.path.join(args.root, "Wildtrack/")
         
        
        intrinsic_camera_matrix_filenames =['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
        extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']
        if isTrain: 
            train_set = PedestrianDataset(args=args, root=dataset_root, org_img_shape=[1080, 1920], world_grid_shape=[480, 1440], transform=transform, intrinsic_camera_matrix_filenames=intrinsic_camera_matrix_filenames, 
                 extrinsic_camera_matrix_filenames=extrinsic_camera_matrix_filenames, is_train=True)
            return train_set
        else: 
            test_set = PedestrianDataset(args=args, root=dataset_root, org_img_shape=[1080, 1920], world_grid_shape=[480, 1440], transform=transform, intrinsic_camera_matrix_filenames=intrinsic_camera_matrix_filenames, 
                 extrinsic_camera_matrix_filenames=extrinsic_camera_matrix_filenames, is_train=False)
            return test_set
    elif 'MultiviewX' in args.dataset:
        dataset_root =os.path.join(args.root, "MultiviewX/")
        intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml']
        extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml']
        if isTrain: 
            train_set = PedestrianDataset(args=args, root=dataset_root, org_img_shape=[1080, 1920], world_grid_shape=[640, 1000], transform=transform, intrinsic_camera_matrix_filenames=intrinsic_camera_matrix_filenames, 
                 extrinsic_camera_matrix_filenames=extrinsic_camera_matrix_filenames, is_train=True)
            return train_set
        else: 
            test_set = PedestrianDataset(args=args, root=dataset_root, org_img_shape=[1080, 1920], world_grid_shape=[640, 1000], transform=transform, intrinsic_camera_matrix_filenames=intrinsic_camera_matrix_filenames, 
                 extrinsic_camera_matrix_filenames=extrinsic_camera_matrix_filenames, is_train=False)
            return test_set
    else: 
        raise Exception('must choose from [Wildtrack, MultiviewX]')






def test(): 
    parser = argparse.ArgumentParser(description='Multiview detector')
     #Dataset 
    parser.add_argument('-r', '--root', type=str, default='../Data/')
    parser.add_argument('-d', '--dataset', type=str, default='Wildtrack', choices=['Wildtrack', 'MultiviewX'])
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--world_grid_reduce', type=int, default=1)
    parser.add_argument('--img_reduce', type=int, default=1)
    parser.add_argument('--num_cams', type=int, default=7)   # 6 for MultiviewX
    parser.add_argument('--num_frames', type=int, default=2000) # 400 for MultiviewXgit 
    parser.add_argument('--train_ratio', type=float, default=0.1)
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('--reID', action='store_true')
    args = parser.parse_args()
    root = '../Data/'
    transform = T.Compose([
        T.ToTensor(),]
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),]
    )
    #train_ratio = 0.9 
    #reID = False
    #intrinsic_camera_matrix_filenames =['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
    #                                'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
    #extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
    #                                 'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']
    train_set = build_dataset(isTrain=True, args=args)
    test_set = build_dataset(isTrain=False, args=args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)



    '''
    for i, (imgs, proj_mats, gt, frame) in enumerate(train_loader):
    
        #exit()
        # viz 
        #if i == 5: 
            #print(i) verified 
            #print(imgs.shape) verified  [1, 7, 3, 1080, 1920]
            #print(proj_mats) # verified
            #exit()
            #print(gt) # verified 
            #print(frame) # verified 
        
            # verified 
            #for j in range(imgs[0].shape[0]): 
            #    pil_img = T.ToPILImage()(imgs[0][j].squeeze())
            #    pil_img.save("image"+str(j)+"_"+str(frame)+".png")
            #exit()
        
        # Viz GT onto img 
        
        # verified
        map_gt = gt['boxes'][0].numpy()
        map_gt[:, 0] = map_gt[:, 0] * 480  
        map_gt[:, 1] = map_gt[:, 1] * 1440  


        proj_mats = proj_mats[0][0].numpy()
        world_grid = map_gt
        world_grid = np.concatenate([world_grid, np.ones([world_grid.shape[0]]).reshape(-1, 1)], axis=1)

        image_coord = proj_mats @ world_grid.T
        img_coord = image_coord[:2, :] / image_coord[2, :]
        img = T.ToPILImage()(imgs[0][0].squeeze())
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()
        img_coord = img_coord.astype(int).transpose()
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for point in img_coord:
            cv2.circle(img, tuple(point.astype(int)), 5, (0, 255, 0), -1)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img.save('img_grid_visualize_test.png')
        
        exit()
    '''
        
    
#test()
