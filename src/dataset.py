import os
import glob
import json
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from src.util import *

## Implement the DataLoader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, data_type="both"):
        self.data_dir_i = os.path.join(data_dir, "images")
        self.data_dir_l = os.path.join(data_dir, "labels")
        self.transform = transform
        self.data_type = data_type

        # Updated at Oct 27 2021
        self.to_tensor = ToTensor()

        if os.path.exists(self.data_dir_i):
            lst_data_i = os.listdir(self.data_dir_i)
            lst_data_i = [f for f in lst_data_i if f.endswith("jpg") | f.endswith("jpeg") | f.endswith("png")]
            lst_data_i.sort()
        else:
            lst_data_i = []

        if os.path.exists(self.data_dir_l):
            lst_data_l = os.listdir(self.data_dir_l)
            for f in lst_data_l:
                if f.endswith("json"):
                    dir_data_l = f
            
            with open(os.path.join(self.data_dir_i, dir_data_l), "r") as json_obj:
                dict_l = json.load(json_obj)
        else:
            dict_l = None
        
        self.lst_data_i = lst_data_i
        self.dict_l = dict_l

    def __len__(self):
        return len(self.lst_data_i)

    def __getitem__(self, index):
        data = {}
        if self.data_type == "image" or self.data_type == "both":
            data_i = plt.imread(os.path.join(self.data_dir_i, self.lst_data_i[index]))[:, :, :3]

            if data_i.ndim == 2:
                data_i = data_i[:, :, np.newaxis]
            if data_i.dtype == np.uint8:
                data_i = data_i / 255.0

            data["input"] = data_i
            data["image"] = data_i

        if self.data_type == "label" or self.data_type == "both":
            l_indexes = np.array(self.dict_l[index]["joints"])
            data_l = np.zeros_like(data_i)
            channels = data_l.shape[-1]
            for channel in channels:
                data_l[l_indexes[0],l_indexes[1],channel] = joints_vis[channel]

            data["hmap"] = data_l

        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)

        return data

class DatasetImages(torch.utils.data.Dataset):
    def __init__(self, data_dir, task=None):
        self.data_dir_i = os.path.join(data_dir, "images")
        
        self.task = task

        if os.path.exists(self.data_dir_i):
            lst_data_i = os.listdir(self.data_dir_i)
            lst_data_i = [f for f in lst_data_i if f.endswith("jpg") | f.endswith("jpeg") | f.endswith("png")]
            lst_data_i.sort()
        else:
            lst_data_i = []
        
        self.lst_data_i = lst_data_i

    def __len__(self):
        return len(self.lst_data_i)

    def __getitem__(self, index):
        data = {}
    
        data_i = plt.imread(os.path.join(self.data_dir_i, self.lst_data_i[index]))[:, :, :3]

        if data_i.ndim == 2:
            data_i = data_i[:, :, np.newaxis]
        if data_i.dtype == np.uint8:
            data_i = data_i / 255.0

        data["image"] = data_i

        return data

class DatasetLabels(torch.utils.data.Dataset):
    def __init__(self, data_dir, task=None):
        self.data_dir_l = os.path.join(data_dir, "labels")

        self.task = task

        if os.path.exists(self.data_dir_l):
            lst_data_l = os.listdir(self.data_dir_l)
            for f in lst_data_l:
                if f.endswith("json"):
                    dir_data_l = f
            
            with open(os.path.join(self.data_dir_l, dir_data_l), "r") as json_obj:
                dict_l = json.load(json_obj)
        else:
            dict_l = None
        
        self.dict_l = dict_l

    def __len__(self):
        return len(self.dict_l)

    def __getitem__(self, index):
        data = {}
        
        data_l = np.array(self.dict_l[index]["joints"])
        data_l = data_l / 255.0

        data_w = np.array(self.dict_l[index]["joints_vis"])

        data["label"] = data_l
        data["weight"] = data_w

        return data

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_a, datasets_b, transform=None):
        self.datasets_a = datasets_a
        self.datasets_b = datasets_b
        self.transform = transform

        self.to_tensor = ToTensor()

    def __getitem__(self, i):
        if self.transform:
            data_a = self.transform(self.datasets_a[i])
            data_b = self.transform(self.datasets_b[i])
            
        data_a = self.to_tensor(data_a)
        data_b = self.to_tensor(data_b)

        return (data_a, data_b)

    def __len__(self):
        return len(self.datasets_a)

## Implement the Transform Functions
class ToTensor(object):
    def __call__(self, data):
        # label, input = data["label"], data["input"]
        #
        # label = label.transpose((2, 0, 1)).astype(np.float32)
        # input = input.transpose((2, 0, 1)).astype(np.float32)
        #
        # data = {"label": torch.from_numpy(label), "input": torch.from_numpy(input)}

        # Updated at Apr 5 2020
        for key, value in data.items():
            if key == "image" or key == "hmap":
                value = value.transpose((2, 0, 1)).astype(np.float32)

            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # label, input = data["label"], data["input"]
        #
        # input = (input - self.mean) / self.std
        # label = (label - self.mean) / self.std
        #
        # data = {"label": label, "input": input}

        # Updated at Apr 5 2020
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data


class RandomFlip(object):
    def __call__(self, data):
        # label, input = data["label"], data["input"]

        ver_flip = np.random.rand() > 0.5
        hor_flip = np.random.rand() > 0.5

        if ver_flip:
            # label = np.fliplr(label)
            # input = np.fliplr(input)

            # Updated at Apr 5 2020
            for key, value in data.items():
                if key == "image" or key == "hmap":
                    data[key] = np.flip(value, axis=0)
                elif key == "label":
                    data[key] = np.abs(1 - value[:, 0])

        if hor_flip:
            # label = np.flipud(label)
            # input = np.flipud(input)

            # Updated at Apr 5 2020
            for key, value in data.items():
                if key == "image" or key == "hmap":
                    data[key] = np.flip(value, axis=1)
                elif key == "label":
                    data[key] = np.abs(1 - value[:, 1])

        # data = {"label": label, "input": input}

        return data

class RandomCrop(object):
  def __init__(self, shape):
      self.shape = shape

  def __call__(self, data):
    # input, label = data["input"], data["label"]
    # h, w = input.shape[:2]

    keys = list(data.keys())

    h, w = data[keys[0]].shape[:2]
    new_h, new_w = self.shape

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
    id_x = np.arange(left, left + new_w, 1)

    # input = input[id_y, id_x]
    # label = label[id_y, id_x]
    # data = {"label": label, "input": input}

    # Updated at Apr 5 2020
    for key, value in data.items():
        if key == "image" or key == "hmap":
            data[key] = value[id_y, id_x]
        elif key == "label":
            empty = np.zeros_like(value)
            empty[..., 0] = value[..., 0] - top
            empty[..., 1] = value[..., 1] - left
            data[key] = empty

    return data

class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1],
                                                    self.shape[2]))

        return data
