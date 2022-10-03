import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
from dataloader import ViPCDataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import params
from decoder.utils.utils import *

from config import params

opt = params()

def farthest_point_sample2(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = torch.device("cuda:0")

    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    centroids_vals = torch.zeros(B, C, npoint).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].clone()
        dist = torch.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[1]  # get the index of the point farthest away
    return centroids, centroids_vals


def mix_shapes_2(X, Y, img):
    """
    combine 2 shapes arbitrarily in each batch.
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    Input:
        X, Y - shape and corresponding labels
    Return:
        mixed shape, labels and proportion
    """
  

    # uniform sampling of points from each shape
    device = torch.device('cuda:0')
    batch_size, _, num_points = X.shape
    index = torch.randperm(batch_size).to(device)  # random permutation of examples in batch
    img2 = img[index]
    # draw lambda from beta distribution
    a = 0.4
    b = 0.6
    lam = (b - a) * np.random.random_sample() + a #np.random.ra #if opt.mixup_params > 0 else 1.0

    num_pts_a = round(lam * num_points)
    num_pts_b = num_points - num_pts_a
   
    X_p = X.permute(0,2,1)#.contiguous()
    pts_vals_a = farthest_point_sample(X_p, num_pts_a)
    pts_vals_b = farthest_point_sample(X_p[index, :], num_pts_b)
    mixed_X = torch.cat((pts_vals_a, pts_vals_b), 1).permute(0,2,1)  # convex combination
    points_perm = torch.randperm(num_points).to(device)  # draw random permutation of points in the shape
    mixed_X = mixed_X[:, :, points_perm]

    Y_p = Y.permute(0,2,1)#.contiguous()
    pts_vals_a_Y = farthest_point_sample(Y_p, num_pts_a)
    pts_vals_b_Y = farthest_point_sample(Y_p[index, :], num_pts_b)
    mixed_Y = torch.cat((pts_vals_a_Y, pts_vals_b_Y), 1).permute(0,2,1)  # convex combination
    mixed_Y = mixed_Y[:, :, points_perm]

    mixed_img = img * lam + img2 * (1-lam)

    return mixed_X, mixed_Y, mixed_img


def mix_shapes(X, Y, img):
    """
    combine 2 shapes arbitrarily in each batch.
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    Input:
        X, Y - shape and corresponding labels
    Return:
        mixed shape, labels and proportion
    """
    ##mixed_X = X.clone()
   # mixed_Y = Y.clone()
   # batch_size, _, num_points = mixed_X.size()
    #mixed_img = torch.ones_like(img).to(device)
    # uniform sampling of points from each shape
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size, _, num_points = X.size()
    index = torch.randperm(batch_size).to(device)  # random permutation of examples in batch
    img2 = img[index]
    # draw lambda from beta distribution
    lam = np.random.beta(1, 1) 

    num_pts_a = round(lam * num_points)
    num_pts_b = num_points - num_pts_a
   

    #print(X.shape)
    _, pts_vals_a = farthest_point_sample2(X, num_pts_a)
    _, pts_vals_b = farthest_point_sample2(X[index, :], num_pts_b)
    mixed_X = torch.cat((pts_vals_a, pts_vals_b), 2)# convex combination
    points_perm = torch.randperm(num_points).to(device)  # draw random permutation of points in the shape
    mixed_X = mixed_X[:, :, points_perm]


  
  
    _, pts_vals_a_Y = farthest_point_sample2(Y, num_pts_a)
    _,pts_vals_b_Y = farthest_point_sample2(Y[index, :], num_pts_b)

    mixed_Y = torch.cat((pts_vals_a_Y, pts_vals_b_Y), 2)  # convex combination
  
    mixed_Y = mixed_Y[:, :, points_perm]
    mixed_img = img * lam + img2 * (1-lam)

    return mixed_X, mixed_Y, mixed_img



def get_pts(pcd):
    points = np.asarray(pcd.points)
    X = []
    Y = []
    Z = []
    for pt in range(points.shape[0]):
        X.append(points[pt][0])
        Y.append(points[pt][1])
        Z.append(points[pt][2])
    return np.asarray(X), np.asarray(Y), np.asarray(Z)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_single_pcd(pcd, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    #pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    X, Y, Z = get_pts(pcd)
    t = Z
    ax.scatter(X, Y, Z, c=t, cmap='inferno', marker='o', s=5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=600)
    plt.close()


def generate_missing_gt(partial, complete, gt_f,  n_c = 1024, n_drop = 1):

    in_points = 2048
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(partial)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    #print(partial.shape)
    #print(complete.shape)
    selected = []
    distances = []
    i = 0
    for curr in complete:
        _, idx, dis = pcd_tree.search_knn_vector_3d(curr, n_drop)
       
        distances.append([i, np.asarray(dis)[0]]) #salvo indice e distanza del NN 
        i+=1

    
    distances.sort(key=lambda row: (row[1]), reverse=True)    
    
    columns = list(zip(*distances))
    #print(np.asarray(columns[0][0:1024]))
    #print(gt_f.shape)
    missing_gt = gt_f[np.asarray(columns[0][0:1024])]

    #print(missing_gt)

    return missing_gt

def generate_missing_gt_v2(partial, complete, n_c = 1024):

    distance = torch.cdist(complete, partial)
    min_d = torch.min(distance, dim=2)[0]
    idx_part = torch.topk(min_d, n_c, 1)[1]
    idx_exp = idx_part.unsqueeze(2).expand(partial.shape[0], n_c, 3)
    missing_gt = torch.gather(complete, 1, idx_exp)



    return missing_gt


if __name__ == '__main__':
    
    opt = params()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ViPCDataset_test = ViPCDataLoader(
        'test_list.txt', data_path=opt.dataroot, status="test", view_align=False, category=opt.cat)
    test_loader = DataLoader(ViPCDataset_test,
                            batch_size=64,
                            num_workers=opt.nThreads,
                            shuffle=True,
                            drop_last=True)

    
    for image, gt, partial in test_loader:  

        image = image.to(device)
        gt = gt.to(device)
        partial = partial.to(device)

        partial = farthest_point_sample(partial, 2048)
        gt = farthest_point_sample(gt, 2048)

        #print(partial.shape)
        #print(gt.shape)

        

        # for m in range(opt.batch_size-1):
            
        #     gt_i = gt[m]
        #     gt_f = gt_i.cpu().numpy()
            
        #     gt_f = gt_f.squeeze()

        #     partial_i = partial[m]
        #     gt_i = gt_i.tolist()

        #     partial_i = partial_i.cpu().numpy()
            
        #     partial_i = partial_i.squeeze()

            
            
            # gt_i = gt_i.cpu().numpy()
    
            # gt_i = gt_i.squeeze()

        missing_gt = generate_missing_gt_v2(partial, gt)
        print(missing_gt.shape)
        #print(missing_gt.shape)
        pcd_par = o3d.geometry.PointCloud()
        pcd_par.points = o3d.utility.Vector3dVector(missing_gt[0,:,:].cpu().numpy().squeeze())
        #o3d.io.write_point_cloud(f"./visualization_examples/part_{opt.cat}_test1.ply", pcd_par)
        plot_single_pcd(pcd_par, f"./visualization_examples/render/missing_{opt.cat}.png" )
        
        pcd_s = o3d.geometry.PointCloud()
        pcd_s.points = o3d.utility.Vector3dVector(gt[0].cpu().numpy().squeeze())
        #o3d.io.write_point_cloud(f"./visualization_examples/part_{opt.cat}_test1.ply", pcd_par)
        plot_single_pcd(pcd_s, f"./visualization_examples/render/gt_{opt.cat}.png" )

        pcd_p = o3d.geometry.PointCloud()
        pcd_p.points = o3d.utility.Vector3dVector(partial[0].cpu().numpy().squeeze())
        #o3d.io.write_point_cloud(f"./visualization_examples/part_{opt.cat}_test1.ply", pcd_par)
        plot_single_pcd(pcd_p, f"./visualization_examples/render/partial_{opt.cat}.png" )
        
        
                
        break


            
    


