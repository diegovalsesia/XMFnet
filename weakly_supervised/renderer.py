import pytorch3d
import torch
from torch import nn as nn
from dataloader import ViPCDataLoader2
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import params
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import random
import os
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings
)

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)


from render.core.cloud import PointClouds3D

def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, black)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, white)
    
    return final_conv

def rgb2gray(rgb):
    return torch.inner(rgb[...,:3], torch.tensor([0.2989, 0.5870, 0.1140]).to(rgb.device))


def create_lights():

    lights = DirectionalLights(device=device)

    return lights


def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True


opt = params()


class Renderer(nn.Module):

    """A class for rendering a batch of points in a differentiable manner.
     It combines rasterization, lightning and compositing. This class is based on Pytorch3D differentiable rendering."""

    def __init__(self, colors):
        super().__init__()
        self.image_size = opt.image_size
        self.radius = opt.radius
        self.points_per_pixel = opt.points_per_pixel
        self.dist = opt.dist
        self.compositor = AlphaCompositor()
        self.colors = colors

    def forward(self, points, eye):

        point_cloud = PointClouds3D(points=points, features=self.colors)
        R, T = look_at_view_transform(eye=eye)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=points.device)
       

        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size, radius=self.radius, points_per_pixel=self.points_per_pixel)
        
        rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer, compositor = self.compositor)

        images = renderer(point_cloud)
        
        return images











if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ViPCDataset_train = ViPCDataLoader(
        'train_list2.txt', data_path=opt.dataroot, status="train", category="plane")
    train_loader = DataLoader(ViPCDataset_train,
                              batch_size=16,
                              num_workers=opt.nThreads,
                              shuffle=True,
                              drop_last=True)
    
    renderer = Renderer().to(device)

    for data in tqdm(train_loader):
       
       
        break
