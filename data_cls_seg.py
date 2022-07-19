#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:10:09 2022

@author: linxi
"""

import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    max_distance = np.sqrt(np.max(np.sum(pc**2, axis=1)))
    pc /= max_distance
    return pc


def get_object_id(x) :    #get all kinds of ObjectID from numpy file

    dic = []
    for i in range(x.shape[0]):
        if x[i][6] not in dic:
            dic.append(x[i][6])

    return dic


class MotorDataset(Dataset):
    def __init__(self, root, split='train', num_points=2048, test_area='Validation',
                  block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.block_size = block_size
        self.transform = transform
        motor_ls = sorted(os.listdir(root))
        if split == 'train':
            motor_ids = [motor for motor in motor_ls if '{}'.format(test_area) not in motor]
        else:
            motor_ids = [motor for motor in motor_ls if '{}'.format(test_area) in motor]
        
        self.motor_cat = set([motor.split('_')[1] for motor in motor_ls])
        self.classes = dict(zip(self.motor_cat, range(len(self.motor_cat))))
        self.all_points = [] 
        self.all_types = [] 
        self.all_labels = [] 
        num_points_eachmotor= []
        label_num_eachtype = np.zeros(7)
        for i in tqdm(motor_ids, total=len(motor_ids)):
            motor_type = self.classes[i.split('_')[1]]
            motor_type = np.array([motor_type]).astype(np.int64)
            motor_data = np.load(os.path.join(root, i))
            point_set = motor_data[:, 0:3]
            motor_labels = motor_data[:, 6]
            num_eachtype_in_onemotor,_ = np.histogram(motor_labels, bins=7, range=(0,7))
            label_num_eachtype += num_eachtype_in_onemotor
            self.all_points.append(point_set)
            self.all_types.append(motor_type)
            self.all_labels.append(motor_labels)
            num_points_eachmotor.append(motor_labels.size)
            
    #############caculate the index for choose of points from the motor according to the number of points of a specific motor######### 
        sample_prob_eachmotor = num_points_eachmotor / np.sum(num_points_eachmotor)
        num_interaction = sample_rate * np.sum(num_points_eachmotor) / self.num_points
        self.motors_index = []
        for index in range(len(num_points_eachmotor)):
            sample_times_in_onemotor = int(round(sample_prob_eachmotor[index] * num_interaction))
            motor_index_onemotor = [index] * sample_times_in_onemotor
            self.motors_index.extend(motor_index_onemotor)
    
    def __len__(self):
        return len(self.motors_index)
    
    def __getitem__(self, index):
        point_set = self.all_points[self.motors_index[index]]
        labels = self.all_labels[self.motors_index[index]]
        types = self.all_types[self.motors_index[index]]
        n_points = point_set.shape[0]
        choose = np.random.choice(n_points, self.num_points, replace=True)
        chosed_points = point_set[choose, :]
        chosed_labels = labels[choose]        
        chosed_points = pc_normalize(chosed_points)
        return chosed_points, chosed_labels, types


class MotorData(Dataset):
    def __init__(self, root, split='train', num_points=2048, test_area=None, transform=None):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.transform = transform
        motor_ls = sorted(os.listdir(root))
        motor_ids = {}
        motor_ids['train'] = [motor for motor in motor_ls if '{}'.format(test_area) not in motor]
        motor_ids['test'] = [motor for motor in motor_ls if '{}'.format(test_area) in motor]
        self.motor_cat = set([motor.split('_')[1] for motor in motor_ls])
        self.classes = dict(zip(self.motor_cat, range(len(self.motor_cat))))
        assert (split == 'train' or split == 'test')
        type_names = [x.split('_')[1] for x in motor_ids[split]]
        self.datapath = [(type_names[i], os.path.join(self.root, motor_ids[split][i])) for i
                          in range(len(motor_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))
        self.all_points = [None] * len(self.datapath)
        self.all_cls = [None] * len(self.datapath)
        self.all_labels = [None] * len(self.datapath)
        for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
            fn = self.datapath[index]
            motor_type = self.classes[self.datapath[index][0]]
            motor_type = np.array([motor_type]).astype(np.int64)
            motor_data = np.load(fn[1])
            point_set = motor_data[:, 0:3]
            labels = motor_data[:, 6]
            self.all_points[index] = point_set
            self.all_cls[index] = motor_type
            self.all_labels[index] = labels
    
    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
        point_set = self.all_points[index]
        labels = self.all_labels[index]
        types = self.all_cls[index]
        n_points = point_set.shape[0]
        choice = np.random.choice(n_points, self.num_points, replace=True)
        point_set = point_set[choice, :]
        point_set = pc_normalize(point_set)
        label = labels[choice]
        return point_set, label, types


if __name__ == '__main__':
    import torch
    data = MotorDataset('E:\\dataset', split='train', test_area='Validation')
    Dataloader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    for point, label, types in Dataloader:
        print(point.size())
        # print(label.size())
        # print(types.size())