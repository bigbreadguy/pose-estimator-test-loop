# Simple Baselines for Human Pose Estimation and Tracking
# https://arxiv.org/abs/1804.06208v2

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from src.evaluate import accuracy

import torch
import torch.nn as nn

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        # size_average and reduce args will be deprecated, using reduction='mean' instead.
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, torch.Tensor), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(nn.functional.softmax(heatmaps_reshaped), 2)
    maxvals = torch.amax(nn.functional.softmax(heatmaps_reshaped), 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = torch.tile(idx, (1, 1, 2))

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = torch.tile(torch.gt(maxvals, 0.0), (1, 1, 2))

    preds *= pred_mask
    return preds, maxvals

class PCKhLoss(nn.Module):
    def __init__(self, thr, hm_type='gaussian'):
        super(PCKhLoss, self).__init__()
        self.thr = thr
        self.hm_type = hm_type
        self.get_max_preds = get_max_preds


    def calc_dists(self, preds, target, normalize):
        """
        pyTorch automatically convert
        integers into single precision floats
        """
        # preds = preds.astype(np.float32)
        # target = target.astype(np.float32)
        dists = torch.zeros((preds.shape[1], preds.shape[0]))
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = torch.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
        return dists


    def dist_acc(self, dists):
        ''' Return percentage below threshold while ignoring values with a -1 '''
        dist_cal = torch.ne(dists, -1)
        num_dist_cal = dist_cal.sum()
        if num_dist_cal > 0:
            return torch.lt(dists[dist_cal], self.thr).sum() * 1.0 / num_dist_cal
        else:
            return -1


    def accuracy(self, output, target):
        '''
        Calculate accuracy according to PCK,
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
        '''
        idx = list(range(output.shape[1]))
        norm = 1.0
        if self.hm_type == 'gaussian':
            pred, _ = self.get_max_preds(batch_heatmaps=output)
            target, _ = self.get_max_preds(batch_heatmaps=target)
            h = output.shape[2]
            w = output.shape[3]
            norm = torch.ones((pred.shape[0], 2)) * torch.Tensor([h, w]) / 10
        dists = self.calc_dists(pred, target, norm)

        acc = torch.zeros((len(idx) + 1))
        avg_acc = 0
        cnt = 0

        for i in range(len(idx)):
            acc[i + 1] = self.dist_acc(dists[idx[i]])
            if acc[i + 1] >= 0:
                avg_acc = avg_acc + acc[i + 1]
                cnt += 1

        avg_acc = avg_acc / cnt if cnt != 0 else 0
        if cnt != 0:
            acc[0] = avg_acc
        return acc, avg_acc, cnt, pred
    
    def forward(self, output, target):
        acc, avg_acc, cnt, pred = self.accuracy(output, target)
        
        del avg_acc, cnt, pred

        return acc