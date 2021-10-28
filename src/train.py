import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.loss import JointsMSELoss
from src.model import *
from src.dataset import *
from src.util import *

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

    norm = args.norm

    network = args.network
    resnet_depth = args.resnet_depth
    joint_weight = args.joint_weight

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
                                              RandomFlip(),
                                              Normalization(mean=MEAN, std=STD)])

        image_train = DatasetImages(data_dir=os.path.join(data_dir, 'train'),
                                    task=task)

        label_train = DatasetLabels(data_dir=os.path.join(data_dir, 'train'),
                                    task=task)

        dataset_train = ConcatDataset(image_train, label_train,
                                      transform=transform_train)

        loader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=NUM_WORKER)
        
        # Set Other Variables 
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

    if network == "PoseResNet":
        netP = PoseResNet(in_channels=nch, out_channels=num_mark, nker=nker, norm=norm, num_layers=resnet_depth).to(device)
        init_weights(netP, init_type='normal', init_gain=0.02)
    
    ## Define the Loss Functions
    fn_pose = JointsMSELoss(use_target_weight=joint_weight).to(device)

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

            for batch, (data_i, data_l) in enumerate(loader_train, 1):
                print(len(data_i))
                print(data_i[1])
                input_data = data_i.to(device)
                pose_label = data_l.to(device)
                target_weight = data_w.to(device)

                # forward netP
                output = netP(input_data)

                # Build target heatmap from pose labels
                target = torch.zeros_like(output)
                target[:, :, pose_label[:,:,0], pose_label[:,:,1]] = 1

                # backward netP
                set_requires_grad(netP, True)
                optimP.zero_grad()

                loss_P = fn_pose(output, target, target_weight)
                loss_P.backward()
                optimP.step()

                # compute the losses
                loss_P_train += [loss_P.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "POSE a %.4f b %.4f | \n"%
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_P_train)))
                
                if batch % 50 == 0:
                    # Save to the Tensorboard
                    input_data = fn_tonumpy(fn_denorm(input_data)).squeeze()
                    output = fn_tonumpy(fn_denorm(output)).squeeze()

                    id = num_batch_train * (epoch - 1) + batch

                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input_data[0],
                               cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0],
                               cmap=cmap)
                    
                    writer_train.add_image('input', input_data, id, dataformats='NHWC')
                    writer_train.add_scalar('loss_P', np.mean(loss_P_train), epoch)

                    if epoch % 10 == 0 or epoch == num_epoch:
                        save(ckpt_dir=ckpt_dir, epoch=epoch,
                            netP=netP, optimP=optimP)

    writer_train.close()

def test(args):
    ## Set Hyperparameters for the Testing
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

    norm = args.norm

    network = args.network
    resnet_depth = args.resnet_depth
    joint_weight = args.joint_weight

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

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## Create Result Directories
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test))

    if mode == 'test':
        transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=MEAN, std=STD)])

        image_test = DatasetImages(data_dir=os.path.join(data_dir, 'test'),
                                    task=task, data_type='both')

        label_test = DatasetLabels(data_dir=os.path.join(data_dir, 'test'),
                                    task=task, data_type='both')

        dataset_test = ConcatDataset(image_test, label_test,
                                     transform=transform_test)

        loader_test = DataLoader(dataset_test,
                                    batch_size=batch_size,
                                    shuffle=False, num_workers=NUM_WORKER)
        
        # Set Other Variables 
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    if network == "PoseResNet":
        netP = PoseResNet(in_channels=nch, out_channels=num_mark, nker=nker, norm=norm, num_layers=resnet_depth).to(device)
        init_weights(netP, init_type='normal', init_gain=0.02)
    
    ## Define the Loss Functions
    fn_pose = JointsMSELoss(use_target_weight=joint_weight).to(device)

    ## Set the Optimizers
    optimP = torch.optim.Adam(netP.parameters(), lr=lr, betas=(0.5, 0.999))

    ## Define Other Functions
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x: (x * STD) + MEAN

    cmap = None

    ## Set SummaryWriter for the Tensorboard
    writer_test = SummaryWriter(log_dir=os.path.join(log_dir, 'test'))

    ## Inference
    st_epoch = 0

    if mode == 'test':
        netP, optimP = load(ckpt_dir=ckpt_dir,
                                netP=netP,
                                optimP=optimP)

        with torch.no_grad():
            netP.eval()

            loss_P = []

            for batch, data in enumerate(loader_test, 1):
                input_data = data["image"].to(device)
                pose_label = data["label"].to(device)
                target_weight = data["label"].to(device)

                # forward netP
                output = netP(input_data)

                # Build target heatmap from pose labels
                target = torch.zeros_like(output)
                target[:, :, pose_label[:,:,0], pose_label[:,:,1]] = 1

                loss = fn_pose(output, target, target_weight)

                # compute the losses
                loss_P += [loss.item()]

                # Save to the Tensorboard
                input_data = fn_tonumpy(fn_denorm(input_data))
                output = fn_tonumpy(fn_denorm(output))
                target = fn_tonumpy(fn_denorm(target))

                for j in range(input_data.shape[0]):
                    id = batch_size * (batch - 1) + j
                    
                    input_data_ = input_data[j]
                    output_ = output[j]
                    target_ = target[j]

                    input_data_ = np.clip(input_data_, a_min=0, a_max=1)
                    output_ = np.clip(output_, a_min=0, a_max=1)
                    target_ = np.clip(target_, a_min=0, a_max=1)

                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_data_)
                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_)
                    plt.imsave(os.path.join(result_dir_test, 'png', '%04d_target.png' % id), target_)
                    writer_test.add_image('input', input_data, id, dataformats='NHWC')
                    writer_test.add_image('output', output, id, dataformats='NHWC')
                    writer_test.add_image('target', target, id, dataformats='NHWC')

                    print("TEST: BATCH %04d / %04d | " % (id + 1, num_data_test))

                writer_test.add_scalar('loss', loss_P[-1], batch)
    
    writer_test.close()