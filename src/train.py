import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import itertools
import matplotlib.pyplot as plt

from torchvision import transforms

MEAN = 0.5
STD = 0.5

NUM_WORKER = 0

def train(args):
    ## Set Hyperparameters for the Training
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    num_mark = args.num_mark

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident
    norm = args.norm

    network = args.network
    resnet_depth = args.resnet_depth

    cuda = args.cuda
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("norm: %s" % norm)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("number of markers: %s" % num_mark)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## Create Result Directories
    result_dir_train = os.path.join(result_dir, 'train')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train))

    if mode == 'train':
        transform_train = transforms.Compose([Resize(shape=(286, 286, nch)),
                                              RandomCrop((ny, nx)),
                                              RandomFlip((ny, nx)),
                                              Normalization(mean=MEAN, std=STD)])

        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'),
                                transform=transform_train,
                                task=task, data_type='both')

        loader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=NUM_WORKER)
        
        # Set Other Variables 
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

    if network == "PoseResnet":
        netP = PoseResNet(in_channels=nch, nker=nker, norm=norm, num_layers=resnet_depth)
        init_weights(netP, init_type='normal', init_gain=0.02)
    
    ## Define the Loss Functions
    fn_pose = nn.MSELoss().to(device)

    ## Set the Optimizers
    optimP = torch.optim.Adam(netP.parameters(), lr=lr, betas=(0.5, 0.999))

    ## Define Other Functions
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x: (x * STD) + MEAN

    cmap = None

    ## Set SummaryWriter for the Tensorboard
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    ## Train the Networks
    st_epoch = 0
    if mode == 'train':
        if train_continue == "on":
            netP, optimP = load(ckpt_dir=ckpt_dir,
                                netP=netP,
                                optimP=optimP)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            netP.train()
            loss_P_train = []

            for batch, data in enumerate(loader_train, 1):
                input_data = data["image"].to(device)
                label = data["label"].to(device)