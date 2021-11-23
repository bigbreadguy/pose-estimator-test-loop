import os
import argparse

import numpy as np
from scipy.stats import poisson
from skimage.transform import rescale, resize

import torch
import torch.nn as nn

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    message = 'initialize network with %s\n' % init_type
    net.apply(init_func)  # apply the initialization function <init_func>

    return message

def save(ckpt_dir, epoch, netP, optimP):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'netP': netP.state_dict(),
                'optimP': optimP.state_dict()},
            "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def load(ckpt_dir, netP, optimP):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        
        return epoch, netP, optimP

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netP.load_state_dict(dict_model['netP'])
    optimP.load_state_dict(dict_model['optimP'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return epoch, netP, optimP

## Add Sampling
def add_sampling(img, type="random", opts=None):
    sz = img.shape

    if type == "uniform":
        ds_y = opts[0].astype(np.int)
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(img.shape)
        msk[::ds_y, ::ds_x, :] = 1

        dst = img * msk

    elif type == "random":
        prob = opts[0]

        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < prob).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd > prob).astype(np.float)

        dst = img * msk

    elif type == "gaussian":
        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]

        a = opts[4]

        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])

        x, y = np.meshgrid(lx, ly)

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd < gaus).astype(np.float)

        # gaus = a * np.exp(-((x - x0) ** 2 / (2 * sgmx ** 2) + (y - y0) ** 2 / (2 * sgmy ** 2)))
        # gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))
        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < gaus).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        dst = img * msk

    return dst

## Add Noise
def add_noise(img, type="random", opts=None):
    sz = img.shape

    if type == "random":
        sgm = opts[0]

        noise = sgm / 255.0 * np.random.randn(sz[0], sz[1], sz[2])

        dst = img + noise

    elif type == "poisson":
        dst = poisson.rvs(255.0 * img) / 255.0
        noise = dst - img

    return dst

## Add blurring
def add_blur(img, type="bilinear", opts=None):
    if type == "nearest":
        order = 0
    elif type == "bilinear":
        order = 1
    elif type == "biquadratic":
        order = 2
    elif type == "bicubic":
        order = 3
    elif type == "biquartic":
        order = 4
    elif type == "biquintic":
        order = 5

    sz = img.shape
    if len(opts) == 1:
        keepdim = True
    else:
        keepdim = opts[1]

    # dw = 1.0 / opts[0]
    # dst = rescale(img, scale=(dw, dw, 1), order=order)
    dst = resize(img, output_shape=(sz[0] // opts[0], sz[1] // opts[0], sz[2]), order=order)

    if keepdim:
        # dst = rescale(dst, scale=(1 / dw, 1 / dw, 1), order=order)
        dst = resize(dst, output_shape=(sz[0], sz[1], sz[2]), order=order)

    return dst


##

def image2patch(src, nimg, npatch, nmargin, datatype="tensor"):
    src = src.to('cpu').detach().numpy()

    nimg_zp = np.zeros(4, np.int32)
    ncrop = np.zeros(4, np.int32)
    nset = np.zeros(4, np.int32)

    for id in range(0, 4):
        nimg_zp[id] = int(nimg[id] + 2 * nmargin[id])
        ncrop[id] = int(npatch[id] - 2 * nmargin[id])
        nset[id] = np.ceil(nimg_zp[id] / ncrop[id]).astype(np.int32)

    nsmp = np.prod(nset)

    iset = [(np.linspace(0, nimg_zp[0] - npatch[0], nset[0])).astype(np.int32),
            (np.linspace(0, nimg_zp[1] - npatch[1], nset[1])).astype(np.int32),
            (np.linspace(0, nimg_zp[2] - npatch[2], nset[2])).astype(np.int32),
            (np.linspace(0, nimg_zp[3] - npatch[3], nset[3])).astype(np.int32)]

    patch = [np.arange(0, npatch[0])[:, np.newaxis, np.newaxis, np.newaxis],
             np.arange(0, npatch[1])[:, np.newaxis, np.newaxis],
             np.arange(0, npatch[2])[:, np.newaxis],
             np.arange(0, npatch[3])]

    src = np.pad(src, ((nmargin[0], nmargin[0]), (nmargin[1], nmargin[1]), (nmargin[2], nmargin[2]), (nmargin[3], nmargin[3])), 'reflect')
    dst = np.zeros((nsmp * npatch[0], npatch[1], npatch[2], npatch[3]), dtype=np.float32)

    for i in range(0, nset[0]):
        for j in range(0, nset[1]):
            for k in range(0, nset[2]):
                for q in range(0, nset[3]):

                    pos = [nset[3] * nset[2] * nset[1] * i + nset[2] * nset[1] * j + nset[1] * k + q]

                    i_ = iset[0][i] + patch[0]
                    j_ = iset[1][j] + patch[1]
                    k_ = iset[2][k] + patch[2]
                    q_ = iset[3][q] + patch[3]

                    dst[pos, :, :, :] = src[i_, j_, k_, q_]

    if datatype == "tensor":
        dst = torch.from_numpy(dst)

    return dst


def patch2image(src, nimg, npatch, nmargin, datatype="tensor", type="count"):
    src = src.to('cpu').detach().numpy()

    nimg_zp = np.zeros(4, np.int32)
    ncrop = np.zeros(4, np.int32)
    nset = np.zeros(4, np.int32)

    for id in range(0, 4):
        nimg_zp[id] = int(nimg[id] + 2 * nmargin[id])
        ncrop[id] = int(npatch[id] - 2 * nmargin[id])
        nset[id] = np.ceil(nimg_zp[id] / ncrop[id]).astype(np.int32)

    nsmp = np.prod(nset)

    iset = [(np.linspace(0, nimg_zp[0] - npatch[0], nset[0])).astype(np.int32),
             (np.linspace(0, nimg_zp[1] - npatch[1], nset[1])).astype(np.int32),
             (np.linspace(0, nimg_zp[2] - npatch[2], nset[2])).astype(np.int32),
             (np.linspace(0, nimg_zp[3] - npatch[3], nset[3])).astype(np.int32)]

    crop = [nmargin[0] + np.arange(0, ncrop[0])[:, np.newaxis, np.newaxis, np.newaxis],
            nmargin[1] + np.arange(0, ncrop[1])[:, np.newaxis, np.newaxis],
            nmargin[2] + np.arange(0, ncrop[2])[:, np.newaxis],
            nmargin[3] + np.arange(0, ncrop[3])]

    dst = np.zeros([nimg_zp[0], nimg_zp[1], nimg_zp[2], nimg_zp[3]], dtype=np.float32)
    wgt = np.zeros([nimg_zp[0], nimg_zp[1], nimg_zp[2], nimg_zp[3]], dtype=np.float32)

    i_img = [np.arange(nmargin[0], nimg_zp[0] - nmargin[0]).astype(np.int32)[:, np.newaxis, np.newaxis, np.newaxis],
             np.arange(nmargin[1], nimg_zp[1] - nmargin[1]).astype(np.int32)[:, np.newaxis, np.newaxis],
             np.arange(nmargin[2], nimg_zp[2] - nmargin[2]).astype(np.int32)[:, np.newaxis],
             np.arange(nmargin[3], nimg_zp[3] - nmargin[3]).astype(np.int32)]

    bnd = [ncrop[0] - iset[0][1] if not len(iset[0]) == 1 else 0,
           ncrop[1] - iset[1][1] if not len(iset[1]) == 1 else 0,
           ncrop[2] - iset[2][1] if not len(iset[2]) == 1 else 0,
           ncrop[3] - iset[3][1] if not len(iset[3]) == 1 else 0]

    if type == 'cos':
        wgt_bnd = [None for _ in range(4)]

        for id in range(1, 4):
            t = np.linspace(np.pi, 2 * np.pi, bnd[id])
            wgt_ = np.ones((ncrop[id]), np.float32)
            wgt_[0:bnd[id]] = (np.cos(t) + 1.0)/2.0

            axis_ = [f for f in range(0, 4)]
            axis_.remove(id)
            wgt_ = np.expand_dims(wgt_, axis=axis_)

            ncrop_ = [ncrop[f] for f in range(0, 4)]
            ncrop_[id] = 1

            wgt_bnd[id] = np.tile(wgt_, ncrop_)

    for i in range(0, nset[0]):
        for j in range(0, nset[1]):
            for k in range(0, nset[2]):
                for q in range(0, nset[3]):

                    wgt_ = np.ones(ncrop, np.float32)

                    if type == 'cos':
                        for id in range(1, 4):
                            if id == 1:
                                axs = j
                            elif id == 2:
                                axs = k
                            elif id == 3:
                                axs = q

                            if axs == 0:
                                wgt_ *= np.flip(wgt_bnd[id], id)
                            elif axs == nset[id] - 1:
                                wgt_ *= wgt_bnd[id]
                            else:
                                wgt_ *= np.flip(wgt_bnd[id], id) * wgt_bnd[id]

                    pos = [nset[3] * nset[2] * nset[1] * i + nset[2] * nset[1] * j + nset[1] * k + q]

                    i_ = iset[0][i] + crop[0]
                    j_ = iset[1][j] + crop[1]
                    k_ = iset[2][k] + crop[2]
                    q_ = iset[3][q] + crop[3]

                    src_ = src[pos, :, :, :]
                    dst[i_, j_, k_, q_] = dst[i_, j_, k_, q_] + src_[crop[0], crop[1], crop[2], crop[3]] * wgt_
                    wgt[i_, j_, k_, q_] = wgt[i_, j_, k_, q_] + wgt_

    if type == 'count':
        dst = dst/wgt

    dst = dst[i_img[0], i_img[1], i_img[2], i_img[3]]
    wgt = wgt[i_img[0], i_img[1], i_img[2], i_img[3]]

    if datatype == "tensor":
        dst = torch.from_numpy(dst)
        wgt = torch.from_numpy(wgt)

    return dst

def pose2image(array):
    shape = array.shape
    if len(shape) == 3:
        intg = array.sum(axis=-1)
        rpt = np.repeat(intg[..., np.newaxis], 3, axis=-1)
        array = rpt

        return array

    elif len(shape) == 4:
        for i in range(shape[0]):
            intg = array[i, ...].sum(axis=-1)
            rpt = np.repeat(intg[..., np.newaxis], 3, axis=-1)
            array = np.repeat(rpt[np.newaxis, ...], shape[0], axis=0)

            return array

def tensor2image(tensor):
    shape = tensor.shape
    if len(shape) == 3:
        intg = tensor.sum(axis=0)
        rpt = intg[None, ...].expand(-1, 3, -1, -1)
        array = rpt

        return array

    elif len(shape) == 4:
        for i in range(shape[0]):
            intg = tensor[i, ...].sum(axis=0)
            rpt = intg[None, ...].expand(-1, 3, -1, -1)
            array = rpt[None, ...].expand(shape[0], -1, -1, -1)

            return array

def reshape2image(array):
    shape = array.shape
    if len(shape)==4:
        if shape[0]==1:
            array = array.reshape((shape[1],shape[2],shape[3]))
            if shape[-1]!=3:
                array = pose2image(array)
            return array
        else:
            result = []
            for i in range(shape[0]):
                arr = array[i,...]
                arr = arr.reshape((shape[1],shape[2],shape[3]))
                if shape[-1]!=3:
                    arr = pose2image(arr)
                result.append(arr[np.newaxis, ...])
            return np.concatenate(tuple(result))

    elif len(shape)==3:
        if shape[-1]!=3:
            array = pose2image(array)
        return array

class Resample(nn.Module):
    def __init__(self):
        super(Resample, self).__init__()
    
    def forward(self, size, target):
        targ_size = target.size()
        ratio_h = targ_size[2] // size[0]
        ratio_w = targ_size[3] // size[1]
        resampled = nn.UpsamplingNearest2d(size=size)(torch.zeros_like(target))
        for i in range(targ_size[0]):
            for j in range(targ_size[1]):
                argmax = torch.argmax(target[i, j, :, :])
                # __floordiv__ a.k.a // operator is now deprecated, torch.div(a, b, rounding_mode="floor") can replace the operator
                resampled[i, j, torch.div(argmax, targ_size[3], rounding_mode="floor").div(ratio_h, rounding_mode="floor"), torch.div(argmax % targ_size[3], ratio_w, rounding_mode="floor")] = 1
        
        return resampled

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

# Early stopping for pytorch
# Originally implemented by https://github.com/Bjarten
# Original implementation https://github.com/Bjarten/early-stopping-pytorch
# Revised to be applied to the test loop
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, ckpt_dir='./checkpoint', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.ckpt_dir = ckpt_dir
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optim, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optim, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} \n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optim, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optim, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ... \n')
            
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        torch.save({'netP': model.state_dict(),
                'optimP': optim.state_dict()},
                "%s/model_epoch%d.pth" % (self.ckpt_dir, epoch))

        self.val_loss_min = val_loss

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="TestLoop",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument("--loop", default="stroll", choices=["stroll", "test"], type=str, dest="loop")

        self.parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
        self.parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")
        self.parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")

        self.parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

        self.parser.add_argument("--task", default="pose estimation", choices=["pose estimation"], type=str, dest="task")

        self.parser.add_argument("--ny", default=256, type=int, dest="ny")
        self.parser.add_argument("--nx", default=256, type=int, dest="nx")
        self.parser.add_argument("--nch", default=3, type=int, dest="nch")
        self.parser.add_argument("--nker", default=64, type=int, dest="nker")

        self.parser.add_argument("--norm", default='inorm', type=str, dest="norm")

        self.parser.add_argument("--network", default="PoseResNet", choices=["PoseResNet"], type=str, dest="network")
        self.parser.add_argument("--resnet_depth", default=50, choices=[18, 34, 50, 101, 152], type=int, dest="resnet_depth")
        self.parser.add_argument("--joint_weight", default=False, type=bool, dest="joint_weight")

        self.parser.add_argument("--cuda", default="cuda", choices=["cuda", "cuda:0", "cuda:1"], type=str, dest="cuda")

        self.parser.add_argument("--spec", default="all", type=str, dest="spec")
    
    def parse(self, args = ""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        return opt