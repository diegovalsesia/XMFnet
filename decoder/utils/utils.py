#from matplotlib import pyplot as plt
#import matplotlib
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import h5py
import random
import time
import os
import math
import numpy as np
from torch.autograd import grad

path = os.path.dirname(os.path.abspath(__file__))+'/'
import sys
sys.path.append(path+'furthestPointSampling/')
import fps
sys.path.append(path+'pyTorchChamferDistance/chamfer_distance/')
from chamfer_distance import ChamferDistance



# ----------------------------------------------------------------------- #
# pytorch or cuda utils
# ----------------------------------------------------------------------- #

def set_seed(seed = 42):
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


def save_ckpt(runner, step):
    save_path = os.path.join(runner.opt.save_path, 'checkpoints')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if runner.epoch % step == 0:
        torch.save({'epoch': runner.epoch, 'state_dict': runner.network.state_dict()}, os.path.join(save_path, 'epoch_%s.pth' % (runner.epoch)))
    if runner.epoch == 1 or runner.cd['overall']['value'] <= runner.best_cd:
        runner.best_cd = runner.cd['overall']['value']
        torch.save({'epoch': runner.epoch, 'state_dict': runner.network.state_dict()}, os.path.join(save_path, 'epoch_best.pth'))


# def preprocess_partial(partial):
#     """
#     remove the duplicated points in partial
#     """
#     partial = np.array(list(set([tuple(t) for t in partial])))
#     return partial


# def preprocess_gt(gt):
#     """
#     remove the duplicated points in gt
#     """
#     gt = np.array(list(set([tuple(t) for t in gt])))
#     return gt


# def resample_pcd(pcd, n):
#     """Drop or duplicate points so that pcd has exactly n points"""
#     idx = np.random.permutation(pcd.shape[0])
#     if idx.shape[0] < n:
#         idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
#     return pcd[idx[:n]]


def farthest_point_sample(xyz, npoints):
    idx = fps.furthest_point_sample(xyz, npoints)
    new_points = fps.gather_operation(xyz.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
    return new_points


# def plot_diff_pcds(pcds, vis, title, legend, win=None):
#     '''
#     :param pcds: python list, include pcds with different size
#     :      legend: each pcds' legend
#     :return:
#     '''
#     device = pcds[0].device
#     assert vis.check_connection()

#     pcds_data = torch.Tensor().to(device)
#     for i in range(len(pcds)):
#         pcds_data = torch.cat((pcds_data, pcds[i]), 0)

#     pcds_label = torch.Tensor().to(device)
#     for i in range(1, len(pcds) + 1):
#         pcds_label = torch.cat((pcds_label, torch.Tensor([i] * pcds[i - 1].shape[0]).to(device)), 0)

#     vis.scatter(X=pcds_data, Y=pcds_label,
#                 opts={
#                     'title': title,
#                     'markersize': 3,
#                     # 'markercolor': np.random.randint(0, 255, (len(pcds), 3)),
#                     'webgl': True,
#                     'legend': legend},
#                 win=win)


# def generate_selfrgb(color_num):
#     """
#     https://www.coder.work/article/3139758
#     """
#     t = np.linspace(-510, 510, color_num)
#     rgb = np.round(np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 255)).astype(np.uint8)
#     # print(rgb, rgb.shape)
#     return rgb


# def generate_cmap(color_num, cmap='viridis'):
#     """
#     https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
#     """
#     viridis = cm.get_cmap(cmap)
#     rgb = np.trunc(viridis(np.linspace(0, 1, color_num))[:,0:3]*255)
#     return rgb  # color_num * 3


# def plot_diff_regions(pcd, vis, title, region_num, win=None):
#     '''
#     :param pcd: one pcd, diff color for diff region
#     :return:
#     '''
#     device = pcd.device
#     assert vis.check_connection()

#     pcd_data = pcd
#     pcd_label = torch.linspace(1, region_num, steps=region_num, dtype=torch.int32).to(device)
#     pcd_label = pcd_label.reshape(region_num, 1).repeat(1, int(pcd.shape[0]/region_num)).reshape(pcd.shape[0])

#     vis.scatter(X=pcd_data, Y=pcd_label,
#                 # update='append',
#                 opts={
#                     'title': title,
#                     'markersize': 3,
#                     'markercolor': generate_cmap(region_num),
#                     'webgl': True,
#                     'legend': ['region'+str(i) for i in range(region_num)]},
#                 win=win)


# def plot_loss_curves(runner, losses, vis, win=None):
#     if not hasattr(runner, 'curve_data'):
#         runner.curve_data = {'X':[], 'Y':[], 'legend':list(losses.keys())}
#     runner.curve_data['X'].append(runner.epoch)
#     runner.curve_data['Y'].append([losses[k] for k in runner.curve_data['legend']])
    
#     vis.line(
#         X=np.array(runner.curve_data['X']),
#         Y=np.array(runner.curve_data['Y']),
#         opts={
#             'title': 'runing loss over time',
#             'legend': runner.curve_data['legend'],
#             'xlabel': 'epoch',
#             'ylabel': 'loss'},
#         win=win)


# ----------------------------------------------------------------------- #
# losses and metrics
# ----------------------------------------------------------------------- #

class L2_ChamferLoss(nn.Module):
    def __init__(self):
        super(L2_ChamferLoss, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        dist = torch.mean(dist1) + torch.mean(dist2)
        return dist


class L2_ChamferEval(nn.Module):
    def __init__(self):
        super(L2_ChamferEval, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        dist = torch.mean(dist1) + torch.mean(dist2)
        return dist * 10000

class L2_ChamferEval_1000(nn.Module):
    def __init__(self):
        super(L2_ChamferEval_1000, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        dist = torch.mean(dist1) + torch.mean(dist2)
        return dist * 1000

class L1_ChamferLoss(nn.Module):
    def __init__(self):
        super(L1_ChamferLoss, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        # print(dist1, dist1.shape) [B, N]
        dist = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
        return dist / 2


class L1_ChamferEval(nn.Module):
    def __init__(self):
        super(L1_ChamferEval, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        dist = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
        return dist / 2 * 1000


class F1Score(nn.Module):
    def __init__(self):
        super(F1Score, self).__init__()
        self.chamfer_dist = ChamferDistance()
    
    def forward(self, array1, array2, threshold=0.001):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
        precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
        fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
        fscore[torch.isnan(fscore)] = 0
        return fscore, precision_1, precision_2




# taken from https://github.com/SymenYang/CPCGAN/blob/main/Model/Gradient_penalty.py
class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, lambdaGP, gamma=1, vertex_num=2500, device=torch.device('cpu')):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.vertex_num = vertex_num
        self.device = device

    def __call__(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        
        fake_data = fake_data[:batch_size]
        
        alpha = torch.rand(batch_size, 1, 1, requires_grad=True).to(self.device)
        # randomly mix real and fake data
        interpolates = real_data + alpha * (fake_data - real_data)
        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates)
        # compute gradients w.r.t the interpolated outputs
        
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous().view(batch_size,-1)
                         
        gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        return gradient_penalty


if __name__ == '__main__':
    from scipy.spatial import cKDTree
    from knn_cuda import KNN

    ref = torch.rand(32, 2048, 3).cuda()
    query = torch.rand(32, 2048, 3).cuda()

    begin = time.time()
    knn = KNN(k=5, transpose_mode=True)
    dist, indx = knn(ref, query)  # B*query*k
    end = time.time()
    shape = indx.shape
    indx = torch.reshape(indx, (shape[0], shape[1]*shape[2]))
    print(dist[0], indx[0])  # dist is from small to large
    print("knn pytorch time: %.4f" % (end-begin))

    np_ref = ref.detach().cpu().numpy()
    np_query = ref.detach().cpu().numpy()

    begin = time.time()
    for b in range(np_ref.shape[0]):
        tree = cKDTree(np_ref[b])
        dist, indx = tree.query(np_query[b], k=32)  # 64 x 32
    end = time.time()
    print("scipy kd-tree time: %.4f" % (end-begin))