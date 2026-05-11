from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_eval_set
from functools import reduce
import scipy.io as sio
import time
import cv2

from JMENet_x4 import Net as JMENet

os.environ["CUDA_VISIBLE_DEVICES"]='1'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=float, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='/data1/A_lOW-Depth/LOW-RGBD-DATA/test_dataset/')
parser.add_argument('--output1', default='/data0/Chen-Bin-Tao/Low-Depth/work/4x/1/Results/', help='Location to save checkpoint models')
parser.add_argument('--output2', default='/data0/Chen-Bin-Tao/Low-Depth/work/4x/1/Results/', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='test_depthLR4/')
parser.add_argument('--test_rgb_dataset', type=str, default='test_color-low')
parser.add_argument('--model_type', type=str, default='JMENet')
parser.add_argument('--model', default="/data0/Chen-Bin-Tao/Low-Depth/work/4x/1/weights/x4/train_depthLR4/amaxJMENet_epoch_150.pth", help='sr pretrained base model')


opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_eval_set(os.path.join(opt.input_dir,opt.test_dataset),os.path.join(opt.input_dir,opt.test_rgb_dataset))
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.model_type == 'JMENet':
    model = JMENet(channel=32)
else:
    model = JMENet(channel=32)
#####
if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

if os.path.exists(opt.model):
    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage),strict=False)
    print('Pre-trained SR model is loaded.<---------------------------->')

if cuda:
    model = model.cuda()

def eval():
    model.eval()
    torch.set_grad_enabled(False)
    for batch in testing_data_loader:
        input,input_rgb, name1,name2 = Variable(batch[0],volatile=True),Variable(batch[1],volatile=True), batch[2], batch[3]
        if cuda:
            input = input.cuda()
            input_rgb = input_rgb.cuda()
        t0 = time.time()
        print('input_rgb',input_rgb.shape,'input',input.shape)
        prec,pre_C,pre_H= model(input_rgb, input)
        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (name1[0], (t1 - t0)))
        save_img1(pre_C.cpu().data, name1[0])
        save_img2(pre_H.cpu().data, name2[0])

def save_img2(img, img_name):

    save_img2 = img.squeeze().clamp(0, 1).numpy()
    import numpy as np
    # save_img=(save_img-np.min(save_img))/(np.max(save_img)-np.min(save_img))
    print(np.min(save_img2*255.0))
    print(np.max(save_img2*255.0))
    save_dir2 = os.path.join(opt.output2, opt.test_dataset)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)

    save_fn2 = save_dir2 +'/'+ img_name
    cv2.imwrite(save_fn2,save_img2*255)

def save_img1(img, img_name):
    assert (len(img.shape) == 4 and img.shape[0] == 1)
    img = img.clone().detach()
    img = img.to(torch.device('cpu'))
    img = img.squeeze()
    img = img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    save_dir1 = os.path.join(opt.output1, opt.test_rgb_dataset)
    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)

    save_fn1 = save_dir1 + '/' + img_name
    cv2.imwrite(save_fn1, img)
##Eval Start!!!!
eval()
