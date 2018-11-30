# -*- coding: utf-8 -*-
"""
load data with pytorch

"""
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from utilties.data_preprocess import normalize_data


class TrafficDataset(Dataset):

    def __init__(self, input_file, transform=None, normalization_flg=False):
        self.X = []
        self.y = []
        with open(input_file, 'r') as fid_in:
            line = fid_in.readline()
            while line:
                line_arr = line.split(',')
                value = list(map(lambda x: float(x), line_arr[:-1]))
                self.X.append(value)
                self.y.append(float(line_arr[-1].strip()))
                line = fid_in.readline()

        if normalization_flg:
            self.X = normalize_data(np.asarray(self.X, dtype=float), range_value=[-1, 1], eps=1e-5)
            with open(input_file+'_normalized.csv', 'w') as fid_out:
                for i in range(self.X.shape[0]):
                    # print('i', i.data.tolist())
                    tmp = [str(j) for j in self.X[i]]
                    fid_out.write(','.join(tmp) + ','+str(int(self.y[i]))+'\n')

        self.transform = transform



    def __getitem__(self, index):

        value_x = self.X[index]
        value_y = self.y[index]
        if self.transform:
            value_x = self.transform(value_x)

        value_x = torch.from_numpy(np.asarray(value_x)).double()
        value_y = torch.from_numpy(np.asarray(value_y)).double()

        # X_train, X_test, y_train, y_test = train_test_split(value_x, value_y, train_size=0.7, shuffle=True)
        return value_x, value_y  # Dataset.__getitem__() should return a single sample and label, not the whole dataset.
        # return value_x.view([-1,1,-1,1]), value_y

    def __len__(self):
        return len(self.X)


class TrafficDataset_Backup(Dataset):

    def __init__(self, input_file, transform=None):
        self.X = []
        self.y = []
        with open(input_file, 'r') as fid_in:
            line = fid_in.readline()
            while line:
                line_arr = line.split(',')
                value = list(map(lambda x: float(x), line_arr[:-1]))
                self.X.append(value)
                self.y.append(float(line_arr[-1].strip()[0]))
                line = fid_in.readline()
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        value_x = self.X[index]
        value_y = self.y[index]
        if self.transform:
            value_x = self.transform(value_x)

        value_x = torch.from_numpy(np.asarray(value_x)).double()
        value_y = torch.from_numpy(np.asarray(value_y)).double()

        return value_x, value_y


class TrafficDataset_Backup1(Dataset):

    def __init__(self, input_file, transform=None):
        self.X = []
        self.y = []
        with open(input_file, 'r') as fid_in:
            line = fid_in.readline()
            while line:
                line_arr = line.split(',')
                # value = list(map(lambda x: float(x), line_arr[:-1]))
                self.X.append(','.join(line_arr[:-1]))
                self.y.append(float(line_arr[-1].strip()[0]))
                line = fid_in.readline()
        self.transform = transform

    def __getitem__(self, index):

        value_x = self.X[index]
        value_y = self.y[index]
        if self.transform:
            value_x = self.transform(value_x)
            # value_y=self.transform(value_y)

        # value_x = torch.from_numpy(np.asarray(value_x)).double()
        # value_y = torch.from_numpy(np.asarray(value_y)).double()

        return value_x, value_y
        # return value_x.view([-1,1,-1,1]), value_y

    def __len__(self):
        return len(self.X)


class Str2List(object):
    def __call__(self, x_str):
        line_arr = x_str.split(',')
        value = list(map(lambda x: float(x), line_arr))

        return value


class List2Tensor(object):

    def __call__(self, value_x):
        value_x = torch.from_numpy(np.asarray(value_x))

        return value_x



def split_train_test(dataset, split_percent=0.7, shuffle=True, seed=42):
    """
    refer: https://stackoverflow.com/questions/50544730/split-dataset-train-and-test-in-pytorch-using-custom-dataset
    :param dataset:
    :param split_percent:
    :param shuffle:
    :param seed:
    :return:
    """
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split_percent * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                            sampler=train_sampler)
    # validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                                 sampler=test_sampler)

    return train_sampler, test_sampler


if __name__ == '__main__':
    input_file = '../data/data_split_train_v2_711/train_1pkt_images_merged.csv'
    demo_flg = True
    if demo_flg:
        data_transform = transforms.Compose([Str2List(),
                                             List2Tensor(),
                                             ])
        traffic_dataset = TrafficDataset_Backup1(input_file, transform=data_transform)
        dataset_loader = torch.utils.data.DataLoader(traffic_dataset,
                                                     batch_size=4,
                                                     shuffle=True,
                                                     num_workers=4)

        for i, (x, y) in enumerate(dataset_loader):
            print(i, x, y)
    # # data_transform = transforms.Compose([
    # #     transforms.ToTensor(),
    # #     transforms.Normalize(mean=[0.485],
    # #                          std=[0.229])
    # # ])
    # dataset = TrafficDataset(input_file, transform=None)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=True, num_workers=4)
    # for step, (b_x, b_y) in enumerate(dataloader):
    #     # print(b_x, b_y)
    #     print(b_x.shape)
    #     tmp_x = b_x.view([b_x.shape[0], 1, -1,
    #                       1])  # For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width
    #     print(tmp_x.shape)
