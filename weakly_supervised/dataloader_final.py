import os
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm
import matplotlib.pyplot as plt
from decoder.utils.utils import *
from decoder.utils.draw import *
import open3d as o3d



import scipy.io as sio

def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, black)
    
    return final_conv


def rgb2gray_np(rgb):
    return np.dot(rgb[...,:3],[0.2989, 0.5870, 0.1140])


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def rotation_z(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                [sin_theta, cos_theta, 0.0],
                                [0.0, 0.0, 1.0]])
    return pts @ rotation_matrix.T


def rotation_y(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                [0.0, 1.0, 0.0],
                                [sin_theta, 0.0, cos_theta]])
    return pts @ rotation_matrix.T


def rotation_x(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, cos_theta, -sin_theta],
                                [0.0, sin_theta, cos_theta]])
    return pts @ rotation_matrix.T

class ViPCDataLoader_ft(Dataset):
    def __init__(self,filepath,data_path,status,pc_input_num=3500, view_align=False, category='all'):
        super(ViPCDataLoader_ft,self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'plane':'02691156',
            'bench': '02828884',  
            'cabinet':'02933112', 
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117', 
            'lamp':'03636649',
            'speaker': '03691459', 
            'firearm': '04090263', 
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088',
            'watercraft':'04530566'
        }
        with open(filepath,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()
        
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')
        self.partpart_path = os.path.join(data_path, 'ShapeNetViPC-PartPart')

        for key in self.filelist:
            if category !='all':
                if key.split(';')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split(';')[0])
            self.key.append(key)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.key)}')


    def __getitem__(self, idx):
        
        key = self.key[idx]
       
        pc_part_path = os.path.join(self.imcomplete_path,key.split(';')[0]+'/'+ key.split(';')[1]+'/'+key.split(';')[-1].replace('\n', '')+'.dat')
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')
        
       
        pc_path = os.path.join(self.gt_path, ran_key.split(';')[0]+'/'+ ran_key.split(';')[1]+'/'+ran_key.split(';')[-1].replace('\n', '')+'.dat')
        view_path = os.path.join(self.rendering_path,ran_key.split(';')[0]+'/'+ran_key.split(';')[1]+'/rendering/'+ran_key.split(';')[-1].replace('\n','')+'.png')
        part_part_path = os.path.join(self.partpart_path,ran_key.split(';')[0]+'/'+ran_key.split(';')[1]+'/'+ran_key.split(';')[-1].replace('\n','')+'.npy')

        part_part_folder = os.path.join(self.partpart_path,ran_key.split(';')[0]+'/'+ran_key.split(';')[1])
        #Inserted to correct a bug in the splitting for some lines 
        if(len(ran_key.split(';')[-1])>3):
            print("bug")
            print(ran_key.split(';')[-1])
            fin = ran_key.split(';')[-1][-2:]
            interm = ran_key.split(';')[-1][:-2]
            print(interm)
            pc_path = os.path.join(self.gt_path, ran_key.split(';')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
            view_path = os.path.join(self.rendering_path,ran_key.split(';')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')


        
        views = self.transform(Image.open(view_path))
       
        views = views[:3,:,:]
        views_t_p = views.permute(1,2,0)
        views_t_p = views_t_p.numpy()
        mask = torch.from_numpy((views_t_p[..., :3].max(-1) != 0).astype(np.float32))
        
        
        # load partial points
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        # increase some item point number less than 3500 
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]


        
        # load the view metadata
        image_view_id = view_path.split('.')[0].split('/')[-1]
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')
        cam_eye = np.loadtxt(view_path[:-6]+'camera_calibration.txt', delimiter='/')
            
        theta_part = math.radians(view_metadata[int(part_view_id),0])
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = rotation_y(rotation_x(pc_part, - phi_part),np.pi + theta_part)
        pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        try:
            with open(part_part_path,'rb') as f:
                pc_partpart = np.load(f)
        except:
            pc_partpart = farthest_point_sample(torch.from_numpy(pc_part).float(), 2048)
    
        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float(), torch.from_numpy(cam_eye).float(), mask, torch.from_numpy(pc_partpart).float()

    def __len__(self):
        return len(self.key)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    category = "plane"
    ViPCDataset = ViPCDataLoader_ft('tester_list.txt',data_path='/home/aiello/ShapeNetViPC-Dataset',status='test', category = category)
    train_loader = DataLoader(ViPCDataset,
                              batch_size=16,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)
    
    


    for image, gt, partial, cam_eye, mask in tqdm(train_loader):
        
    
        break
    