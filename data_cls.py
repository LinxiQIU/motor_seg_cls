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
    def __init__(self, root, split='train', num_points=4096, test_area=None,
                  block_size=1.0, transform=None):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.block_size = block_size
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
            point_set = np.load(fn[1])[:, 0:3]
            # point_set = pc_normalize(point_set)
            self.all_points[index] = point_set
            self.all_cls[index] = motor_type
    
    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
        point_set = self.all_points[index]
        label = self.all_cls[index]
        n_points = point_set.shape[0]
        choice = np.random.choice(n_points, self.num_points, replace=True)
        point_set = point_set[choice, :]
        point_set = pc_normalize(point_set)
        return point_set, label


if __name__ == '__main__':
    import torch
    data = MotorDataset('E:\\dataset', split='train', test_area='Validation')
    Dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    for point, label in Dataloader:
        print(point.shape)
        print(label)
        
        
        
        
            
            
            
        
        
    

