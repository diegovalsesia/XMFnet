import torch
import os
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
from decoder.utils.utils import *
from model import Network
from config import params
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import ViPCDataLoader
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn




opt = params()

if opt.cat != None:

    CLASS = opt.cat
else:
    CLASS = 'plane'


MODEL = 'model_supervised'
FLAG = 'train'
DEVICE = 'cuda:0'
VERSION = '0.1'
BATCH_SIZE = int(opt.batch_size)
MAX_EPOCH = int(opt.n_epochs)
EVAL_EPOCH = int(opt.eval_epoch)
RESUME = False


TIME_FLAG = time.asctime(time.localtime(time.time()))
CKPT_RECORD_FOLDER = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/record'
CKPT_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt.pth'
CONFIG_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/CONFIG.txt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def save_record(epoch, prec1, net: nn.Module):
    state_dict = net.state_dict()
    torch.save(state_dict,
               os.path.join(CKPT_RECORD_FOLDER, f'epoch{epoch}_{prec1:.4f}.pth'))


def save_ckpt(epoch, net, optimizer_all):
    ckpt = dict(
        epoch=epoch,
        model=net.state_dict(),
        optimizer_all=optimizer_all.state_dict(),
    )
    torch.save(ckpt, CKPT_FILE)


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


def weights_init_normal(m):
    """ Weights initialization with normal distribution.. Xavier """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train_one_step(data, optimizer, network):

    image = data[0].to(device)
    partial = data[2].to(device)
    gt = data[1].to(device)

    partial = farthest_point_sample(partial, 2048)
    gt = farthest_point_sample(gt, 2048)

    partial = partial.permute(0, 2, 1)
    network.train()
    complete = network(partial, image)
   
    loss_total = loss_cd(complete, gt)
    
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    return loss_total


best_loss = 99999
best_epoch = 0
resume_epoch = 0
board_writer = SummaryWriter(
    comment=f'{MODEL}_{VERSION}_{BATCH_SIZE}_{FLAG}_{CLASS}_{TIME_FLAG}')

model = Network().apply(weights_init_normal)
loss_cd = L1_ChamferLoss()
loss_cd_eval = L2_ChamferEval()
optimizer = torch.optim.Adam(filter(
    lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999))

ViPCDataset_train = ViPCDataLoader(
    'train_list2.txt', data_path=opt.dataroot, status="train", category=opt.cat)
train_loader = DataLoader(ViPCDataset_train,
                          batch_size=opt.batch_size,
                          num_workers=opt.nThreads,
                          shuffle=True,
                          drop_last=True)

ViPCDataset_test = ViPCDataLoader(
    'test_list2.txt', data_path=opt.dataroot, status="test", category=opt.cat)
test_loader = DataLoader(ViPCDataset_test,
                         batch_size=opt.batch_size,
                         num_workers=opt.nThreads,
                         shuffle=True,
                         drop_last=True)


if RESUME:
    ckpt_path = "./model_path/model.pt"
    ckpt_dict = torch.load(ckpt_path)
    model.load_state_dict(ckpt_dict['model_state_dict'])
    optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
    resume_epoch = ckpt_dict['epoch']
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


if not os.path.exists(os.path.join(CKPT_RECORD_FOLDER)):
    os.makedirs(os.path.join(CKPT_RECORD_FOLDER))

with open(CONFIG_FILE, 'w') as f:
    f.write('RESUME:'+str(RESUME)+'\n')
    f.write('FLAG:'+str(FLAG)+'\n')
    f.write('DEVICE:'+str(DEVICE)+'\n')
    f.write('BATCH_SIZE:'+str(BATCH_SIZE)+'\n')
    f.write('MAX_EPOCH:'+str(MAX_EPOCH)+'\n')
    f.write('CLASS:'+str(CLASS)+'\n')
    f.write('VERSION:'+str(VERSION)+'\n')
    f.write(str(opt.__dict__))


model.train()
model.to(device)

print('--------------------')
print('Training Starting')
print(f'Training Class: {CLASS}')
print('--------------------')

set_seed()

for epoch in range(resume_epoch, resume_epoch + opt.n_epochs+1):

    if epoch < 25:
        opt.lr = 0.001
    elif epoch < 120:
        opt.lr = 0.0001
    else:
        opt.lr = 0.00001

    Loss = 0
    i = 0

    for data in tqdm(train_loader):

        loss = train_one_step(data, optimizer, network=model)
        i += 1
        if i % opt.loss_print == 0:
            board_writer.add_scalar("Loss_iteration", loss.item(
            ), global_step=i + epoch * len(train_loader))
            
        Loss += loss

    Loss = Loss/i
    print(f"epoch {epoch}: Loss = {Loss}")
    board_writer.add_scalar("Average_Loss_epochs", Loss.item(), epoch)

    if epoch % EVAL_EPOCH == 0: 
        
        with torch.no_grad():
            model.eval()
            i = 0
            Loss = 0
            for data in tqdm(test_loader):

                i += 1
                image = data[0].to(device)
                partial = data[2].to(device)
                gt = data[1].to(device)

                partial = farthest_point_sample(partial, 2048)
                gt = farthest_point_sample(gt, 2048)

                partial = partial.permute(0, 2, 1)

                complete = model(partial, image)

                loss = loss_cd_eval(complete, gt)
                
                Loss += loss

            Loss = Loss/i
            board_writer.add_scalar(
                "Average_Loss_epochs_test", Loss.item(), epoch)

            if Loss < best_loss:
                best_loss = Loss
                best_epoch = epoch
            print(best_epoch, ' ', best_loss)

    print('****************************')
    print(best_epoch, ' ', best_loss)
    print('****************************')

    if epoch % opt.ckp_epoch == 0:  

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': Loss
        }, f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt_{epoch}.pt')


print('Train Finished!!')
