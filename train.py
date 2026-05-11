from __future__ import print_function
import argparse
from math import log10
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import pdb
import socket
import time
import scipy.io as scio
import Myloss


from JMENet_x4 import Net as JMENet

os.environ["CUDA_VISIBLE_DEVICES"]='0'


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=10, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=float, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/data1/A_lOW-Depth/LOW-RGBD-DATA/train_dataset/')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='train_depth/')
parser.add_argument('--RGB_train_dataset', type=str, default='train_color/')
parser.add_argument('--rgb_train_dataset', type=str, default='train_color-low/')
parser.add_argument('--train_dataset', type=str, default='train_depthLR4/')
parser.add_argument('--model_type', type=str, default='JMENet')
parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='xxx.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='./weights/x4/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='', help='Location to save checkpoint models')
opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
print(opt)
SSIM_loss = Myloss.SSIM1()
# Gaussian = Gaussian_c(3).cuda()
def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input_rgb, input, target1, target2= Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
        if cuda:
            input_rgb = input_rgb.cuda()
            input = input.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()

        optimizer.zero_grad()
        t0 = time.time()
        pre_c, pre_C,pre_H=model(input_rgb, input)

        loss_c0 = criterion_l2(pre_C, target1) + 0.1 * (1 - SSIM_loss(pre_C, target1))
        loss_c1 = criterion_l2(pre_c, target1) + 0.1 * (1 - SSIM_loss(pre_c, target1))

        loss_c = loss_c0 + loss_c1

        loss_h0 = criterion(pre_H, target2)

        loss_h = loss_h0

        loss = 0.1 * loss_c + loss_h

        t1 = time.time()
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        avg_loss = epoch_loss / len(training_data_loader)
        loss_list.append(avg_loss)

        print("===> Epoch[{}]({}/{}): Loss: {:.5f} || Timer: {:.5f} sec.".format(epoch, iteration, len(training_data_loader), loss.item(), (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.5f}".format(epoch, epoch_loss / len(training_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+opt.train_dataset+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.train_dataset, opt.hr_train_dataset, opt.rgb_train_dataset, opt.RGB_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)


training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)
if opt.model_type == 'JMENet':
    model = JMENet(channel=32)
else:
    model = JMENet(channel=32)

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()
criterion_l2 = nn.MSELoss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        pretrained_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print('***********')


        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('************************************Pre-trained SR model is loaded.************************************')
###############
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    # model = model.cuda(gpus_list[1])
    # criterion = criterion.cuda(gpus_list[1])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)


loss_list = []
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)

    if (epoch+1) == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch+1) == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)


