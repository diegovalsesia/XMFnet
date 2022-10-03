import torch
import numpy as np
import open3d as o3d
import randpartial

from visdom import Visdom


def plot_diff_pcds(pcds, vis, title, legend, win=None):
    '''
    :param pcds: python list, include pcds with different size
    :      legend: each pcds' legend
    :return:
    '''
    device = pcds[0].device
    assert vis.check_connection()

    pcds_data = torch.Tensor().to(device)
    for i in range(len(pcds)):
        pcds_data = torch.cat((pcds_data, pcds[i]), 0)

    pcds_label = torch.Tensor().to(device)
    for i in range(1, len(pcds) + 1):
        pcds_label = torch.cat((pcds_label, torch.Tensor([i] * pcds[i - 1].shape[0]).to(device)), 0)

    vis.scatter(X=pcds_data, Y=pcds_label,
                opts={
                    'title': title,
                    'markersize': 3,
                    # 'markercolor': np.random.randint(0, 255, (len(pcds), 3)),
                    'webgl': True,
                    'legend': legend},
                win=win)


if __name__ == "__main__":
    vis = Visdom(env='randpartial')

    pcd1 = o3d.io.read_point_cloud('./1a04e3eab45ca15dd86060f189eb133.pcd')
    pcd1 = torch.from_numpy(np.asarray(pcd1.points))

    pcd2 = torch.from_numpy(randpartial.gen(pcd1))
    plot_diff_pcds([pcd1, pcd2], vis=vis, title='test', legend=['pcd1', 'pcd2'], win='test')