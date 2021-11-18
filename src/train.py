import os
from datetime import datetime, timedelta, timezone, date

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.loss import JointsMSELoss
from src.model import *
from src.dataset import *
from src.util import *
from src.evaluate import *

import itertools
import matplotlib.pyplot as plt

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

    log_prefix = args.log_prefix

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


    ## Open log file and write
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    f = open(log_prefix + "-" + mode + ".txt", "a")
    f.write("initiate %s loop : " % mode + date_time + "\n")

    f.write("mode: %s\n" % mode)
    f.write("norm: %s\n" % norm)

    f.write("learning rate: %.4e\n" % lr)
    f.write("batch size: %d\n" % batch_size)
    f.write("number of epoch: %d\n" % num_epoch)
    
    f.write("task: %s\n" % task)
    f.write("number of markers: %s\n" % num_mark)

    f.write("network: %s\n" % network)

    f.write("data dir: %s\n" % data_dir)
    f.write("ckpt dir: %s\n" % ckpt_dir)
    f.write("log dir: %s\n" % log_dir)
    f.write("result dir: %s\n" % result_dir)

    f.write("device: %s\n" % device)

    ## Create Result Directories
    result_dir_train = os.path.join(result_dir, 'train')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train))

    if mode == 'train':
        transform_train = "3R1N" # Resize - RandomCrop - RandomFlip - Normalization

        dataset_full = Dataset(data_dir=os.path.join(data_dir, 'train'),
                                transform=transform_train, shape=(ny, nx, nch), hm_shape=(ny, nx, num_mark))
        
        # Set Other Variables 
        num_data = len(dataset_full)
        num_data_train = num_data // 10 * 9
        num_batch_train = np.ceil(num_data_train / batch_size)

        dataset_train, dataset_val = torch.utils.data.random_split(dataset_full, [num_data_train, num_data-num_data_train])

        loader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=NUM_WORKER)

        loader_val = DataLoader(dataset_val,
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=NUM_WORKER)

    if network == "PoseResNet":
        netP = PoseResNet(in_channels=nch, out_channels=num_mark, nker=nker, norm=norm, num_layers=resnet_depth).to(device)
        message = init_weights(netP, init_type='normal', init_gain=0.02)
        f.write(message)
    
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
            st_epoch, netP, optimP = load(ckpt_dir=ckpt_dir,
                                netP=netP,
                                optimP=optimP)
        
        early_stop = EarlyStopping(ckpt_dir=ckpt_dir)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            netP.train()
            loss_P_train = []
            val_data = next(iter(loader_val))
            val_input = val_data["image"].to(device)
            val_target = val_data["hmap"]

            for batch, data in enumerate(loader_train, 1):
                input_data = data["image"].to(device)
                target = data["hmap"].to(device)
                target_weight = None

                # forward netP
                output = netP(input_data)

                # Build target heatmap from pose labels
                # try interpolation - deprecated
                # target = nn.functional.interpolate(target, (output.size()[1], output.size()[2], output.size()[3]), mode="nearest")
                scale_factor = (output.size()[2]/target.size()[2], output.size()[3]/target.size()[3])
                resample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
                target = resample(target)

                # backward netP
                set_requires_grad(netP, True)
                optimP.zero_grad()

                loss_P = fn_pose(output, target)
                loss_P.backward()
                optimP.step()

                # compute the losses
                loss_P_train += [float(loss_P.item())]

                f.write("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "POSE LOSS %.4f | \n"%
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_P_train)))
                
                if batch % 50 == 0:
                    # Save to the Tensorboard
                    input_data = fn_tonumpy(fn_denorm(input_data)).squeeze()
                    output = fn_tonumpy(fn_denorm(output)).squeeze()

                    input_data = np.clip(input_data, a_min=0, a_max=1)

                    # Convert pose heatmap into image form
                    output = pose2image(output)
                    output = np.clip(output, a_min=0, a_max=1)

                    id = num_batch_train * (epoch - 1) + batch

                    if not batch_size==1:
                        plt.imsave(os.path.join(result_dir_train, '%04d_input.png' % id), input_data[0],
                                cmap=cmap)
                        plt.imsave(os.path.join(result_dir_train, '%04d_output.png' % id), output[0],
                                cmap=cmap)
                        writer_train.add_image('input', input_data, id, dataformats='NHWC')
                        writer_train.add_image('output', input_data, id, dataformats='NHWC')
                    else:
                        plt.imsave(os.path.join(result_dir_train, '%04d_input.png' % id), input_data,
                                cmap=cmap)
                        plt.imsave(os.path.join(result_dir_train, '%04d_output.png' % id), output,
                                cmap=cmap)
                        writer_train.add_image('input', input_data, id, dataformats='HWC')
                        writer_train.add_image('output', input_data, id, dataformats='HWC')
                    writer_train.add_scalar('loss_P', np.mean(loss_P_train), epoch)

                    if epoch % 10 == 0 or epoch == num_epoch:
                        save(ckpt_dir=ckpt_dir, epoch=epoch,
                            netP=netP, optimP=optimP)

            # forward netP
            with torch.no_grad():
                netP.eval()
                val_output = netP(val_input)
                val_target = nn.functional.interpolate(val_target, (val_output.size()[2], val_output.size()[3]), mode="nearest").to(device)
                
                # Early stop when validation loss does not reduce
                val_loss = fn_pose(val_output, val_target, None)
                early_stop(val_loss=val_loss, model=netP, optim=optimP, epoch=epoch, trace_func=f.write)
                
            if early_stop.early_stop:
                break

    writer_train.close()
    f.close()

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

    log_prefix = args.log_prefix

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

    ## Open log file and write
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    f = open(log_prefix + "-" + mode + ".txt", "a")
    f.write("initiate %s loop : " % mode + date_time + "\n")

    f.write("mode: %s\n" % mode)
    f.write("norm: %s\n" % norm)

    f.write("learning rate: %.4e\n" % lr)
    f.write("batch size: %d\n" % batch_size)
    f.write("number of epoch: %d\n" % num_epoch)

    f.write("task: %s\n" % task)
    f.write("number of markers: %s\n" % num_mark)

    f.write("network: %s\n" % network)

    f.write("data dir: %s\n" % data_dir)
    f.write("ckpt dir: %s\n" % ckpt_dir)
    f.write("log dir: %s\n" % log_dir)
    f.write("result dir: %s\n" % result_dir)

    f.write("device: %s\n" % device)

    ## Create Result Directories
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test))

    if mode == 'test':
        transform_test = "RN" # Resize - Normalization

        dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'),
                                     transform=transform_test, shape=(ny, nx, nch), hm_shape=(ny, nx, num_mark))

        loader_test = DataLoader(dataset_test,
                                    batch_size=batch_size,
                                    shuffle=False, num_workers=NUM_WORKER)
        
        # Set Other Variables 
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    if network == "PoseResNet":
        netP = PoseResNet(in_channels=nch, out_channels=num_mark, nker=nker, norm=norm, num_layers=resnet_depth).to(device)
        message = init_weights(netP, init_type='normal', init_gain=0.02)
        f.write(message)
    
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
        epoch, netP, optimP = load(ckpt_dir=ckpt_dir,
                                netP=netP,
                                optimP=optimP)

        with torch.no_grad():
            netP.eval()

            loss_P = []

            for batch, data in enumerate(loader_test, 1):
                input_data = data["image"].to(device)
                target = data["hmap"].to(device)
                target_weight = None

                # forward netP
                output = netP(input_data)

                # Build target heatmap from pose labels
                # try interpolation - deprecated
                # target = nn.functional.interpolate(target, (output.size()[1], output.size()[2], output.size()[3]), mode="nearest")

                scale_factor = (output.size()[2]/target.size()[2], output.size()[3]/target.size()[3])
                resample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
                target = resample(target)

                loss = fn_pose(output, target)

                # compute the losses
                loss_P += [float(loss.item())]

                # Save to the Tensorboard
                input_data = fn_tonumpy(fn_denorm(input_data))
                output = fn_tonumpy(fn_denorm(output))
                target = fn_tonumpy(fn_denorm(target))

                if not batch_size==1:
                    for j in range(input_data.shape[0]):
                        id = batch_size * (batch - 1) + j
                        
                        input_data_ = input_data[j]
                        output_ = output[j]
                        target_ = target[j]

                        input_data_ = np.clip(input_data_, a_min=0, a_max=1)
                        
                        # Convert pose heatmaps into image form
                        output_ = reshape2image(output_)
                        output_ = np.clip(output_, a_min=0, a_max=1)
                        target_ = reshape2image(target_)
                        target_ = np.clip(target_, a_min=0, a_max=1)

                        plt.imsave(os.path.join(result_dir_test, '%04d_input.png' % id), input_data_)
                        plt.imsave(os.path.join(result_dir_test, '%04d_output.png' % id), output_)
                        plt.imsave(os.path.join(result_dir_test, '%04d_target.png' % id), target_)
                        writer_test.add_image('input', input_data, id, dataformats='NHWC')
                        writer_test.add_image('output', output_, id, dataformats='NHWC')
                        writer_test.add_image('target', target_, id, dataformats='NHWC')

                        f.write("TEST: BATCH %04d / %04d | POSE LOSS %.8f | \n" % (id + 1, num_data_test, np.mean(loss_P)))
                else:
                    id = batch_size * (batch - 1) + 0
                        
                    input_data_ = input_data
                    output_ = output
                    target_ = target

                    input_data_ = np.clip(input_data_, a_min=0, a_max=1)

                    # Convert pose heatmaps into image form
                    output_ = reshape2image(output_)
                    output_ = np.clip(output_, a_min=0, a_max=1)
                    target_ = reshape2image(target_)
                    target_ = np.clip(target_, a_min=0, a_max=1)

                    plt.imsave(os.path.join(result_dir_test, '%04d_input.png' % id), input_data_)
                    plt.imsave(os.path.join(result_dir_test, '%04d_output.png' % id), output_)
                    plt.imsave(os.path.join(result_dir_test, '%04d_target.png' % id), target_)
                    writer_test.add_image('input', input_data, id, dataformats='HWC')
                    writer_test.add_image('output', output_, id, dataformats='HWC')
                    writer_test.add_image('target', target_, id, dataformats='HWC')

                    f.write("TEST: BATCH %04d / %04d | POSE LOSS %.8f | \n" % (id + 1, num_data_test, np.mean(loss_P)))

                writer_test.add_scalar('loss', loss_P[-1], batch)
    
    writer_test.close()
    f.close()

def evaluate(args):
    ## Set Hyperparameters for the Evaluation
    mode = "test"

    lr = args.lr
    batch_size = args.batch_size

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir

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

    if mode == 'test':
        dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'),
                                     transform=None, shape=(ny, nx, nch), hm_shape=(ny, nx, num_mark))

        loader_test = DataLoader(dataset_test,
                                    batch_size=batch_size,
                                    shuffle=False, num_workers=NUM_WORKER)

    if network == "PoseResNet":
        netP = PoseResNet(in_channels=nch, out_channels=num_mark, nker=nker, norm=norm, num_layers=resnet_depth).to(device)
        message = init_weights(netP, init_type='normal', init_gain=0.02)
        del message
    
    ## Define the Loss Functions
    fn_pose = JointsMSELoss(use_target_weight=joint_weight).to(device)

    ## Set the Optimizers
    optimP = torch.optim.Adam(netP.parameters(), lr=lr, betas=(0.5, 0.999))

    ## Define Other Functions
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
    fn_denorm = lambda x: (x * STD) + MEAN

    ## Inference

    if mode == 'test':
        epoch, netP, optimP = load(ckpt_dir=ckpt_dir,
                                netP=netP,
                                optimP=optimP)

        with torch.no_grad():
            netP.eval()

            evals = []

            for batch, data in enumerate(loader_test, 1):
                input_data = data["image"].to(device)
                target = data["hmap"].to(device)
                target_weight = None

                # forward netP
                output = netP(input_data)

                # Build target heatmap from pose labels
                scale_factor = (output.size()[2]/target.size()[2], output.size()[3]/target.size()[3])
                resample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
                target = resample(target)

                # Convert tensors to numpy arrays
                input_data = fn_tonumpy(fn_denorm(input_data))
                output = fn_tonumpy(fn_denorm(output))
                target = fn_tonumpy(fn_denorm(target))

                if not batch_size==1:
                    for j in range(input_data.shape[0]):                        
                        output_ = output[j]
                        target_ = target[j]
                        
                        acc, avg_acc, cnt, pred = accuracy(output_, target_)

                        evals.append({"acc" : acc.tolist(), "avg_acc" : avg_acc,
                                      "cnt" : cnt, "pred" : pred.tolist()})

                else:
                    output_ = output
                    target_ = target

                    acc, avg_acc, cnt, pred = accuracy(output_, target_)

                    evals.append({"acc" : acc.tolist(), "avg_acc" : avg_acc,
                                  "cnt" : cnt, "pred" : pred.tolist()})
    
    return evals