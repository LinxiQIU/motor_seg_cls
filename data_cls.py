# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 02:50:17 2022

@author: linux
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


class MotorDataset(Dataset):
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
        for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
            fn = self.datapath[index]
            motor_type = self.classes[self.datapath[index][0]]
            motor_type = np.array([motor_type]).astype(np.int64)
            motor_data = np.load(fn[1])
            point_set = motor_data[:, 0:3]
            # point_set = pc_normalize(point_set)
            self.all_points[index] = point_set
            self.all_cls[index] = motor_type

    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
        point_set = self.all_points[index]
        types = self.all_cls[index]
        n_points = point_set.shape[0]
        choice = np.random.choice(n_points, self.num_points, replace=True)
        point_set = point_set[choice, :]
        point_set = pc_normalize(point_set)
        return point_set, types


class MotorData(Dataset):
    def __init__(self, root, split='train', num_points=2048, test_area='Validation',
                 sample_rate=1.0):
        super().__init__()
        self.root = root
        self.num_points = num_points
        motor_ls = sorted(os.listdir(root))
        if split == 'train':
            motor_ids = [motor for motor in motor_ls if '{}'.format(test_area) not in motor]
        else:
            motor_ids = [motor for motor in motor_ls if '{}'.format(test_area) in motor]
        
        self.motor_cat = set([motor.split('_')[1] for motor in motor_ls])
        self.classes = dict(zip(self.motor_cat, range(len(self.motor_cat))))        
        self.all_points = [] 
        self.all_types = [] 
        num_points_all= []
        # label_num_eachtype = np.zeros(7)
        for ids in tqdm(motor_ids, total=len(motor_ids)):
            motor_type = self.classes[ids.split('_')[1]]
            motor_type = np.array([motor_type]).astype(np.int64)
            motor_data = np.load(os.path.join(root, ids))     # xyzrgbl, N*7
            point_set = motor_data[:, 0:3]     # xyz
            motor_labels = motor_data[:, 6]       # label
            # tmp, _ = np.histogram(motor_labels, range(8))
            # label_num_eachtype += tmp
            self.all_points.append(point_set)
            self.all_types.append(motor_type)
            num_points_all.append(motor_labels.size)
            
        sample_prob = num_points_all / np.sum(num_points_all)
        # print(sample_prob)
        num_inter = sample_rate * np.sum(num_points_all) / self.num_points
        self.motor_idxs = []
        for idx in range(len(num_points_all)):
            sample_times_in_onemotor = int(round(sample_prob[idx] * num_inter))
            motor_index_onemotor = [idx] * sample_times_in_onemotor
            self.motor_idxs.extend(motor_index_onemotor)
            # print(self.motor_idxs)
    
    def __len__(self):
        return len(self.motor_idxs)
    
    def __getitem__(self, index):
        point_set = self.all_points[self.motor_idxs[index]]
        # labels = self.all_labels[self.motors_index[index]]
        types = self.all_types[self.motor_idxs[index]]
        n_points = point_set.shape[0]
        choose = np.random.choice(n_points, self.num_points, replace=True)
        chosed_points = point_set[choose, :]
        # chosed_labels = labels[choose]        
        chosed_points = pc_normalize(chosed_points)
        return chosed_points, types
        
            

if __name__ == '__main__':
    import torch
    data = MotorDataset(root='E:\\dataset', split='test', test_area='Validation')
    Dataloader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False)
    for points, types in Dataloader:
        print(points.shape)
        print(types.shape)
        
        
        
        
            
            
            
        
        
    

