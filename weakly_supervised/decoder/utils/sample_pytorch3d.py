"""
ShapeNet dataset download link: http://shapenet.cs.stanford.edu/shapenet/obj-zip/
"""

import argparse
import torch
import torch.utils.data as data
import os
import h5py
import pandas as pd
from tqdm import tqdm
from visdom import Visdom

from pytorch3d.datasets import (
    ShapeNetCore,
    collate_batched_meshes,
)
from pytorch3d.renderer import (
    TexturesVertex,
)
from pytorch3d.structures import (
    Meshes, 
    Pointclouds,
)
from pytorch3d.ops import (
    sample_points_from_meshes,
)
from pytorch3d.io import (
    load_obj,
)

from utils import plot_diff_pcds, farthest_point_sample


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

"""
shapenet_dataset = ShapeNetCore("../data/ShapeNetCore.v2", synsets=["02691156"], version=2)

# batch_size = 5
# loader = data.DataLoader(shapenet_dataset, batch_size=batch_size, collate_fn=collate_batched_meshes)

vis = Visdom(env='sample')
df = pd.read_csv("../data/ShapeNetCore.v2/all.csv")

for i in range(len(shapenet_dataset)):
    shapenet_model = shapenet_dataset[i]
    print(i)
    foldername = shapenet_model["synset_id"]
    filename = shapenet_model["model_id"]
    model_verts, model_faces = shapenet_model["verts"], shapenet_model["faces"]
    
    # model_textures = TexturesVertex(verts_features=torch.ones_like(model_verts, device=device)[None])
    shapenet_model_mesh = Meshes(
        verts=[model_verts.to(device)],
        faces=[model_faces.to(device)],
        # textures=model_textures
    )
    samples, normals = sample_points_from_meshes(shapenet_model_mesh, num_samples=16384, return_normals=True)
    # print(samples.shape)
    samples = farthest_point_sample(samples, 2048)

    B, N = samples.shape[:2]
    samples_mean = samples.mean(axis=1).reshape(B, 1, 3)
    scale = torch.max(torch.max(samples - samples_mean, dim=2, keepdim=True)[0], dim=1, keepdim=True)[0].repeat(B, 1, 3)
    samples = (samples - samples_mean) * 0.5 / scale

    # plot_diff_pcds([samples[0]], vis=vis, title=foldername+'_'+filename, legend=['pcd'], win=foldername+'_'+filename)
    samples = samples[0]

    _df = df.loc[df['synsetId'] == int(foldername)]
    split = _df.loc[_df['modelId'] == filename].iloc[0, 4]

    path = os.path.join('../data/ShapeNetCore.v2.PC2048', foldername, split)
    if not os.path.exists(path):
        os.makedirs(path)

    with h5py.File(os.path.join(path, filename+'.h5'), 'w') as f:
        f.create_dataset('data', data=samples.detach().cpu().numpy())
"""


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="../data/ShapeNetCore.v2", help="path to ShapeNetCore.v2")
parser.add_argument('--output_path', type=str, default="../data/ShapeNetCore.v2.PC2048_1", help="path to the sampling results")
parser.add_argument('--synsetid', type=str, default="02691156", help="")
parser.add_argument('--pnum', type=int, default=2048, help="point number, no more than 16384")
args = parser.parse_args()


# vis = Visdom(env='sample')
df = pd.read_csv(os.path.join(args.input_path, 'all.csv'))
_df = df.loc[df['synsetId'] == int(args.synsetid)]

for i in tqdm(range(len(_df))): 
    filename = _df.iloc[i, 3]
    split =_df.iloc[i, 4]
    
    path = os.path.join(args.output_path, args.synsetid, split)
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, filename+'.h5')
    if os.path.exists(save_path):
        continue
    
    path = os.path.join(args.input_path, args.synsetid, filename, 'models/model_normalized.obj')
    if not os.path.exists(path):
        continue
    verts, faces, aux = load_obj(path, load_textures=False, device=device)
    shapenet_model_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

    samples, normals = sample_points_from_meshes(shapenet_model_mesh, num_samples=16384, return_normals=True)
    # print(samples.shape)
    samples = farthest_point_sample(samples, args.pnum)

    B, N = samples.shape[:2]
    samples_mean = samples.mean(axis=1).reshape(B, 1, 3)
    scale = torch.max(torch.max(torch.abs(samples - samples_mean), dim=2, keepdim=True)[0], dim=1, keepdim=True)[0].repeat(B, 1, 3)
    samples = (samples - samples_mean) * 0.5 / scale

    # plot_diff_pcds([samples[0]], vis=vis, title=foldername+'_'+filename, legend=['pcd'], win=foldername+'_'+filename)
    # if i == 20:
    #     break

    samples = samples[0]
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('data', data=samples.detach().cpu().numpy())