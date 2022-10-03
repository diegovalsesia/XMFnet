cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar'])]

import numpy as np
import open3d as o3d

from matplotlib import pyplot as plt
#from utils import *


# ----------------------------------------------------------------------- #
# matplotlib utils
# ----------------------------------------------------------------------- #

def draw_diff_pcds(filename, pcds, titles=None, suptitle='', sizes=None, cmap=['Reds'], zdir='y',
                        xlim=(-0.32, 0.32), ylim=(-0.32, 0.32), zlim=(-0.32, 0.32)):  # ShapeNetCore.v2.PC2048
                        # xlim=(-0.25, 0.25), ylim=(-0.25, 0.25), zlim=(-0.25, 0.25)):  # PCN
    if sizes is None:
        sizes = [5 for i in range(len(pcds))]
        # sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 3))
    for i in range(1):
        elev = 30  # which height to view the surface
        azim = 180-(-45 + 90 * i)  # angle of rotation
        # azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            ax = fig.add_subplot(1, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=-pcd[:, 0].astype(int), s=size, cmap=cmap[j], vmin=-1, vmax=0.5)
            # ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=pcd[:, 0], s=size, cmap=cmap[j], vmin=-1, vmax=0.5)
            if titles is not None:
                ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()


def draw_diff_regions(filename, pcds, region_num, titles=None, suptitle='', sizes=None, cmap=['Reds'], zdir='y',
                        xlim=(-0.32, 0.32), ylim=(-0.32, 0.32), zlim=(-0.32, 0.32)):  # ShapeNetCore.v2.PC2048
    if sizes is None:
        sizes = [5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 3))
    for i in range(1):
        elev = 30
        azim = 180-(-45 + 90 * i)
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            ax = fig.add_subplot(1, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            step = int(pcd.shape[0]/region_num)
            for k in range(region_num):
                data = pcd[k*step:(k+1)*step, :]
                ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=-data[:, 0], s=sizes[j], cmap=cmap[k], vmin=-1, vmax=0.5)
            if titles is not None:
                ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()


def draw_loss_curves_subplot(filename, loss):
    epoch = np.arange(1, 200+1)
    linewidth = 2
    
    fig = plt.figure(figsize=(12, 8))
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)

    ax1.plot(epoch, loss[0], label='FC-based', linewidth=linewidth)
    ax1.plot(np.arange(200-len(loss[1])+1, 200+1), loss[1], label='Folding-based', linewidth=linewidth)
    ax1.plot(epoch, loss[2], label='Ours', linewidth=linewidth)
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('(a) Airplane')

    ax2.plot(epoch, loss[3], label='FC-based', linewidth=linewidth)
    ax2.plot(np.arange(200-len(loss[4])+1, 200+1), loss[4], label='Folding-based', linewidth=linewidth)
    ax2.plot(epoch, loss[5], label='Ours', linewidth=linewidth)
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('(b) All')
    
    if filename is not None:
        fig.savefig(filename, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def multi_pcds_draw(filename, paths, cmap):
    output_dir = filename[::-1].split('/', 1)[-1][::-1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num = len(paths)
    pcds = []
    for i in range(num):
        data = np.asarray(o3d.io.read_point_cloud(paths[i]).points)
        pcds.append(data)
    draw_diff_pcds(filename, pcds, cmap=cmap)


def multi_regions_draw(filename, paths, cmap, region_num=4):
    output_dir = filename[::-1].split('/', 1)[-1][::-1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num = len(paths)
    pcds = []
    for i in range(num):
        data = np.asarray(o3d.io.read_point_cloud(paths[i]).points)
        pcds.append(data)
    draw_diff_regions(filename, pcds, region_num=region_num, cmap=cmap)


if __name__ == "__main__":
    # ----------------------------------------------------------------------- #
    # introduction and approach
    # ----------------------------------------------------------------------- #

    # paths = [
    #         # "../data/PCN/ShapeNet/train/partial/02691156/1a04e3eab45ca15dd86060f189eb133/00.pcd",
    #         "../data/PCN/ShapeNet/train/gt/02691156/1a04e3eab45ca15dd86060f189eb133.pcd",
    #         ]
    # cmap = ['Reds']
    # multi_pcds_draw('../output/1.png', paths, cmap)
    
    # for path, dir_list, file_list in os.walk("../output/axformnet/approach/src"):
    #     for file_name in file_list:
    #         data_path = os.path.join(path, file_name)
    #         filename = os.path.join("../output/axformnet/approach", "imgs_raw", file_name.split('.')[0]+'.png')
    #         multi_pcds_draw(filename, [data_path], ['Reds'])

    # # 3, 4, 12
    # multi_pcds_draw("../output/axformnet/approach/imgs/partial.png", ["../output/axformnet/approach/src/partial.pcd"], ['Blues'])
    # multi_pcds_draw("../output/axformnet/approach/imgs/gt.png", ["../output/axformnet/approach/src/gt.pcd"], ['Reds'])
    # multi_pcds_draw("../output/axformnet/approach/imgs/coarse.png", ["../output/axformnet/approach/src/coarse.pcd"], ['Purples'])
    # multi_pcds_draw("../output/axformnet/approach/imgs/fine.png", ["../output/axformnet/approach/src/fine.pcd"], ['Greens'])

    # multi_pcds_draw("../output/axformnet/approach/imgs/coarse_3.png", ["../output/axformnet/approach/src/coarse_3.pcd"], ['Purples'])
    # multi_pcds_draw("../output/axformnet/approach/imgs/coarse_4.png", ["../output/axformnet/approach/src/coarse_4.pcd"], ['Purples'])
    # multi_pcds_draw("../output/axformnet/approach/imgs/coarse_12.png", ["../output/axformnet/approach/src/coarse_12.pcd"], ['Purples'])
    # multi_pcds_draw("../output/axformnet/approach/imgs/fine_3.png", ["../output/axformnet/approach/src/fine_3.pcd"], ['Greens'])
    # multi_pcds_draw("../output/axformnet/approach/imgs/fine_4.png", ["../output/axformnet/approach/src/fine_4.pcd"], ['Greens'])
    # multi_pcds_draw("../output/axformnet/approach/imgs/fine_12.png", ["../output/axformnet/approach/src/fine_12.pcd"], ['Greens'])


    # ----------------------------------------------------------------------- #
    # training loss
    # ----------------------------------------------------------------------- #

    # # airplane
    # './log/fc_folding/2021-08-06T11:39:10.216082/runlog.txt'
    # './log/fc_folding/2021-08-06T11:40:44.674864/runlog.txt'
    # './log/axform/2021-08-06T11:43:20.259197/runlog.txt'

    # train_loss1 = []
    # with open('./log/fc_folding/2021-08-06T11:39:10.216082/runlog.txt') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if '[80/89]Loss:' in line:
    #              train_loss1.append(float(line.split(' ')[-3]))
    # # print(train_loss1, len(train_loss1))
    # train_loss2 = []
    # with open('./log/fc_folding/2021-08-06T11:40:44.674864/runlog.txt') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if '[80/89]Loss:' in line:
    #              train_loss2.append(float(line.split(' ')[-3]))
    # train_loss3 = []
    # with open('./log/axform/2021-08-06T11:43:20.259197/runlog.txt') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if '[80/89]Loss:' in line:
    #              train_loss3.append(float(line.split(' ')[-3]))
    # train_loss_airplane = [train_loss1, train_loss2[125:], train_loss3]

    # # all
    # './log/fc_folding/2021-08-07T10:00:09.132071/runlog.txt'
    # './log/fc_folding/2021-08-07T10:00:53.608474/runlog.txt'
    # './log/axform/2021-08-07T10:01:19.567676/runlog.txt'

    # train_loss1 = []
    # with open('./log/fc_folding/2021-08-07T10:00:09.132071/runlog.txt') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if '[300/310]Loss:' in line:
    #              train_loss1.append(float(line.split(' ')[-3]))
    # # print(train_loss1, len(train_loss1))
    # train_loss2 = []
    # with open('./log/fc_folding/2021-08-07T10:00:53.608474/runlog.txt') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if '[300/310]Loss:' in line:
    #              train_loss2.append(float(line.split(' ')[-3]))
    # train_loss3 = []
    # with open('./log/axform/2021-08-07T10:01:19.567676/runlog.txt') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if '[300/310]Loss:' in line:
    #              train_loss3.append(float(line.split(' ')[-3]))
    # train_loss_all = [train_loss1, train_loss2[50:], train_loss3]

    # train_loss = train_loss_airplane + train_loss_all
    # draw_loss_curves_subplot('../output/axform/reconstruction/loss_training.png', train_loss)
    

    # ----------------------------------------------------------------------- #
    # one branch
    # ----------------------------------------------------------------------- #

    # paths = [
    #         "../output/axform/reconstruction/one_branch/src/02691156/fc.pcd",
    #         "../output/axform/reconstruction/one_branch/src/02691156/folding.pcd",
    #         "../output/axform/reconstruction/one_branch/src/02691156/axform.pcd",
    #         "../output/axform/reconstruction/one_branch/src/02691156/gt.pcd",
    #         ]
    # cmap = ['Blues', 'Blues', 'Blues', 'Reds']
    # multi_pcds_draw('../output/axform/reconstruction/one_branch/imgs/airplane.png', paths, cmap)

    # paths = [
    #         "../output/axform/reconstruction/one_branch/src/02958343/fc.pcd",
    #         "../output/axform/reconstruction/one_branch/src/02958343/folding.pcd",
    #         "../output/axform/reconstruction/one_branch/src/02958343/axform.pcd",
    #         "../output/axform/reconstruction/one_branch/src/02958343/gt.pcd",
    #         ]
    # cmap = ['Blues', 'Blues', 'Blues', 'Reds']
    # multi_pcds_draw('../output/axform/reconstruction/one_branch/imgs/car.png', paths, cmap)

    # paths = [
    #         "../output/axform/reconstruction/one_branch/src/03001627/fc.pcd",
    #         "../output/axform/reconstruction/one_branch/src/03001627/folding.pcd",
    #         "../output/axform/reconstruction/one_branch/src/03001627/axform.pcd",
    #         "../output/axform/reconstruction/one_branch/src/03001627/gt.pcd",
    #         ]
    # cmap = ['Blues', 'Blues', 'Blues', 'Reds']
    # multi_pcds_draw('../output/axform/reconstruction/one_branch/imgs/chair.png', paths, cmap)


    # ----------------------------------------------------------------------- #
    # k branches
    # ----------------------------------------------------------------------- #

    # paths = [
    #         "../output/axform/reconstruction/k_branches/src/02691156/atlasnet.pcd",
    #         # "../output/axform/reconstruction/k_branches/src/02691156/msn.pcd",
    #         "../output/axform/reconstruction/k_branches/src/02691156/axform.pcd",
    #         "../output/axform/reconstruction/k_branches/src/02691156/gt.pcd",
    #         ]
    # cmap = ['Blues', 'Blues', 'Reds']
    # multi_pcds_draw('../output/axform/reconstruction/k_branches/imgs/airplane.png', paths, cmap)

    # paths = [
    #         "../output/axform/reconstruction/k_branches/src/02958343/atlasnet.pcd",
    #         # "../output/axform/reconstruction/k_branches/src/02958343/msn.pcd",
    #         "../output/axform/reconstruction/k_branches/src/02958343/axform.pcd",
    #         "../output/axform/reconstruction/k_branches/src/02958343/gt.pcd",
    #         ]
    # cmap = ['Blues', 'Blues', 'Reds']
    # multi_pcds_draw('../output/axform/reconstruction/k_branches/imgs/car.png', paths, cmap)

    # paths = [
    #         "../output/axform/reconstruction/k_branches/src/03001627/atlasnet.pcd",
    #         # "../output/axform/reconstruction/k_branches/src/03001627/msn.pcd",
    #         "../output/axform/reconstruction/k_branches/src/03001627/axform.pcd",
    #         "../output/axform/reconstruction/k_branches/src/03001627/gt.pcd",
    #         ]
    # cmap = ['Blues', 'Blues', 'Reds']
    # multi_pcds_draw('../output/axform/reconstruction/k_branches/imgs/chair.png', paths, cmap)

    
    # ----------------------------------------------------------------------- #
    # unsupervised semantic segmentation
    # ----------------------------------------------------------------------- #

    paths = [
            "../output/axform/uss/src/02691156/1cb757280b862ae52c7575c9089791ff.pcd",
            "../output/axform/uss/src/02691156/1e0a24e1135e75a831518807a840c4f4.pcd",
            "../output/axform/uss/src/02691156/2a2caad9e540dcc687bf26680c510802.pcd",
            "../output/axform/uss/src/02691156/2c1fff0653854166e7a636089598229.pcd",
            "../output/axform/uss/src/02691156/3b2a19d782234467f9cc1fc25372199f.pcd",
            "../output/axform/uss/src/02691156/3fa511e1882e41eeca8607f540cc62ba.pcd",
            ]
    cmap=['Blues']*3 + ['Blues']*3 + ['Greens'] + ['Reds']*9
    multi_regions_draw('../output/axform/uss/imgs/airplane.png', paths, cmap, region_num=16)
    
    paths = [
            # "../output/axform/uss/src/02958343/1abeca7159db7ed9f200a72c9245aee7.pcd",
            "../output/axform/uss/src/02958343/1be075751d7cfbf9ee8e9bd690a19ec1.pcd",
            # "../output/axform/uss/src/02958343/1cb95c00d3bf6a3a58dbdf2b5c6acfca.pcd",
            # "../output/axform/uss/src/02958343/1e0ada2b1891ea39e79e3bf25d5c768e.pcd",
            "../output/axform/uss/src/02958343/2a82a66ce6273dce601c8ebc794de3f4.pcd",
            "../output/axform/uss/src/02958343/2acbb7959e6388236d068062d5d5809b.pcd",
            # "../output/axform/uss/src/02958343/2b9cebe9ceae3f79186bed5098d348af.pcd",
            "../output/axform/uss/src/02958343/3b56b3bd4f874de23781057335c8a2e8.pcd",
            "../output/axform/uss/src/02958343/3c685bf24a135262e88791d6267b8a1a.pcd",
            "../output/axform/uss/src/02958343/4a1b48e1b53cb6547a3295b198e908bf.pcd",
            ]
    cmap=['Blues']*4 + ['Greens']*1 + ['Reds']*11
    multi_regions_draw('../output/axform/uss/imgs/car.png', paths, cmap, region_num=16)

    paths = [
            # "../output/axform/uss/src/03001627/1ac6531a337de85f2f7628d6bf38bcc4.pcd",
            "../output/axform/uss/src/03001627/1b5e876f3559c231532a8e162f399205.pcd",
            "../output/axform/uss/src/03001627/1b81441b7e597235d61420a53a0cb96d.pcd",
            # "../output/axform/uss/src/03001627/1c758127bc4fdb18be27e423fd45ffe7.pcd",
            "../output/axform/uss/src/03001627/2b6cbad4ba1e9a0645881d7eab1353ba.pcd",
            "../output/axform/uss/src/03001627/2bd045838a2282ab5205884f75aba3a.pcd",
            # "../output/axform/uss/src/03001627/3e8ad99691e8ea4c504721639e19f609.pcd",
            # "../output/axform/uss/src/03001627/4a0b61d33846824ab1f04c301b6ccc90.pcd",
            "../output/axform/uss/src/03001627/4bc5a889b3ef967b9de7cc399bc9b2b3.pcd",
            "../output/axform/uss/src/03001627/6df1ecffaa0abdbf327289c00b6dc9ca.pcd",
            ]
    cmap=['Blues']*4 + ['Greens']*5 + ['Reds']*7
    multi_regions_draw('../output/axform/uss/imgs/chair.png', paths, cmap, region_num=16)


    # ----------------------------------------------------------------------- #
    # point cloud completion
    # ----------------------------------------------------------------------- #

    # paths = [
    #         "../output/axformnet/comparison/src/02691156/partial.pcd",
    #         "../output/axformnet/comparison/src/02691156/pcn.pcd",
    #         "../output/axformnet/comparison/src/02691156/msn.pcd",
    #         "../output/axformnet/comparison/src/02691156/grnet.pcd",
    #         "../output/axformnet/comparison/src/02691156/sparenet.pcd",
    #         "../output/axformnet/comparison/src/02691156/pmpnet.pcd",
    #         "../output/axformnet/comparison/src/02691156/ours_vinilla.pcd",
    #         "../output/axformnet/comparison/src/02691156/ours.pcd",
    #         "../output/axformnet/comparison/src/02691156/gt.pcd",
    #         ]
    # cmap = ['Blues'] + ['Purples']*5 + ['Greens']*2 + ['Reds']
    # multi_pcds_draw('../output/axformnet/comparison/imgs/airplane.png', paths, cmap)

    # paths = [
    #         "../output/axformnet/comparison/src/02933112/partial.pcd",
    #         "../output/axformnet/comparison/src/02933112/pcn.pcd",
    #         "../output/axformnet/comparison/src/02933112/msn.pcd",
    #         "../output/axformnet/comparison/src/02933112/grnet.pcd",
    #         "../output/axformnet/comparison/src/02933112/sparenet.pcd",
    #         "../output/axformnet/comparison/src/02933112/pmpnet.pcd",
    #         "../output/axformnet/comparison/src/02933112/ours_vinilla.pcd",
    #         "../output/axformnet/comparison/src/02933112/ours.pcd",
    #         "../output/axformnet/comparison/src/02933112/gt.pcd",
    #         ]
    # cmap = ['Blues'] + ['Purples']*5 + ['Greens']*2 + ['Reds']
    # multi_pcds_draw('../output/axformnet/comparison/imgs/cabinet.png', paths, cmap)

    # # need 0.25 -> 0.27
    # paths = [
    #         "../output/axformnet/comparison/src/02958343/partial.pcd",
    #         "../output/axformnet/comparison/src/02958343/pcn.pcd",
    #         "../output/axformnet/comparison/src/02958343/msn.pcd",
    #         "../output/axformnet/comparison/src/02958343/grnet.pcd",
    #         "../output/axformnet/comparison/src/02958343/sparenet.pcd",
    #         "../output/axformnet/comparison/src/02958343/pmpnet.pcd",
    #         "../output/axformnet/comparison/src/02958343/ours_vinilla.pcd",
    #         "../output/axformnet/comparison/src/02958343/ours.pcd",
    #         "../output/axformnet/comparison/src/02958343/gt.pcd",
    #         ]
    # cmap = ['Blues'] + ['Purples']*5 + ['Greens']*2 + ['Reds']
    # multi_pcds_draw('../output/axformnet/comparison/imgs/car.png', paths, cmap)

    # paths = [
    #         "../output/axformnet/comparison/src/03001627/partial.pcd",
    #         "../output/axformnet/comparison/src/03001627/pcn.pcd",
    #         "../output/axformnet/comparison/src/03001627/msn.pcd",
    #         "../output/axformnet/comparison/src/03001627/grnet.pcd",
    #         "../output/axformnet/comparison/src/03001627/sparenet.pcd",
    #         "../output/axformnet/comparison/src/03001627/pmpnet.pcd",
    #         "../output/axformnet/comparison/src/03001627/ours_vinilla.pcd",
    #         "../output/axformnet/comparison/src/03001627/ours.pcd",
    #         "../output/axformnet/comparison/src/03001627/gt.pcd",
    #         ]
    # cmap = ['Blues'] + ['Purples']*5 + ['Greens']*2 + ['Reds']
    # multi_pcds_draw('../output/axformnet/comparison/imgs/chair.png', paths, cmap)

    # paths = [
    #         "../output/axformnet/comparison/src/03636649/partial.pcd",
    #         "../output/axformnet/comparison/src/03636649/pcn.pcd",
    #         "../output/axformnet/comparison/src/03636649/msn.pcd",
    #         "../output/axformnet/comparison/src/03636649/grnet.pcd",
    #         "../output/axformnet/comparison/src/03636649/sparenet.pcd",
    #         "../output/axformnet/comparison/src/03636649/pmpnet.pcd",
    #         "../output/axformnet/comparison/src/03636649/ours_vinilla.pcd",
    #         "../output/axformnet/comparison/src/03636649/ours.pcd",
    #         "../output/axformnet/comparison/src/03636649/gt.pcd",
    #         ]
    # cmap = ['Blues'] + ['Purples']*5 + ['Greens']*2 + ['Reds']
    # multi_pcds_draw('../output/axformnet/comparison/imgs/lamp.png', paths, cmap)

    # paths = [
    #         "../output/axformnet/comparison/src/04256520/partial.pcd",
    #         "../output/axformnet/comparison/src/04256520/pcn.pcd",
    #         "../output/axformnet/comparison/src/04256520/msn.pcd",
    #         "../output/axformnet/comparison/src/04256520/grnet.pcd",
    #         "../output/axformnet/comparison/src/04256520/sparenet.pcd",
    #         "../output/axformnet/comparison/src/04256520/pmpnet.pcd",
    #         "../output/axformnet/comparison/src/04256520/ours_vinilla.pcd",
    #         "../output/axformnet/comparison/src/04256520/ours.pcd",
    #         "../output/axformnet/comparison/src/04256520/gt.pcd",
    #         ]
    # cmap = ['Blues'] + ['Purples']*5 + ['Greens']*2 + ['Reds']
    # multi_pcds_draw('../output/axformnet/comparison/imgs/sofa.png', paths, cmap)

    # paths = [
    #         "../output/axformnet/comparison/src/04379243/partial.pcd",
    #         "../output/axformnet/comparison/src/04379243/pcn.pcd",
    #         "../output/axformnet/comparison/src/04379243/msn.pcd",
    #         "../output/axformnet/comparison/src/04379243/grnet.pcd",
    #         "../output/axformnet/comparison/src/04379243/sparenet.pcd",
    #         "../output/axformnet/comparison/src/04379243/pmpnet.pcd",
    #         "../output/axformnet/comparison/src/04379243/ours_vinilla.pcd",
    #         "../output/axformnet/comparison/src/04379243/ours.pcd",
    #         "../output/axformnet/comparison/src/04379243/gt.pcd",
    #         ]
    # cmap = ['Blues'] + ['Purples']*5 + ['Greens']*2 + ['Reds']
    # multi_pcds_draw('../output/axformnet/comparison/imgs/table.png', paths, cmap)

    # paths = [
    #         "../output/axformnet/comparison/src/04530566/partial.pcd",
    #         "../output/axformnet/comparison/src/04530566/pcn.pcd",
    #         "../output/axformnet/comparison/src/04530566/msn.pcd",
    #         "../output/axformnet/comparison/src/04530566/grnet.pcd",
    #         "../output/axformnet/comparison/src/04530566/sparenet.pcd",
    #         "../output/axformnet/comparison/src/04530566/pmpnet.pcd",
    #         "../output/axformnet/comparison/src/04530566/ours_vinilla.pcd",
    #         "../output/axformnet/comparison/src/04530566/ours.pcd",
    #         "../output/axformnet/comparison/src/04530566/gt.pcd",
    #         ]
    # cmap = ['Blues'] + ['Purples']*5 + ['Greens']*2 + ['Reds']
    # multi_pcds_draw('../output/axformnet/comparison/imgs/vessel.png', paths, cmap)