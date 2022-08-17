"""
"""

import os
import numpy as np
import random
from numpy.random import choice
from tqdm import tqdm           #used to display the circulation position, to see where the code is running at
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


####### normalize the point cloud############
def pc_normalize(pc):
    centroid=np.mean(pc,axis=0)
    pc=pc-centroid
    max_distance=np.sqrt(np.max(np.sum(pc**2, axis=1)))
    pc=pc/max_distance
    return pc


def get_object_id(x) :    #get all kinds of ObjectID from numpy file

    dic = []
    for i in range(x.shape[0]):
        if x[i][6] not in dic:
            dic.append(x[i][6])

    return dic


def densify_blots(patch_motor):
    add = []
    for i in range(len(patch_motor)):
        if (patch_motor[i][6]==6) or (patch_motor[i][6] == 5):
            add.append(patch_motor[i])
    add = np.array(add)
    twonn = NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(add[:,0:3])
    _,indices=twonn.kneighbors(add[:,0:3])
    inter = []
    for i in range(indices.shape[0]):
        interpolation = np.zeros(7)
        interpolation[3:7] = add[0][3:7]
        #if the bolt points are closest to eachonter
        if(indices[indices[i][1]][1]==i):
            interpolation[0:3]=add[i][0:3]+(add[indices[i][1]][0:3]-add[i][0:3])/3
            inter.append(interpolation)
        else:
            interpolation[0:3]=add[i][0:3]+(add[indices[i][1]][0:3]-add[i][0:3])/2
            inter.append(interpolation)
    patch_motor=np.concatenate((patch_motor,inter),axis=0)
    return patch_motor


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

        self.all_points = []
        self.all_labels = [] 
        num_points_eachmotor= []
        label_num_eachtype = np.zeros(7)
        for i in tqdm(motor_ids, total=len(motor_ids)):
            motor_data = np.load(os.path.join(root, i))
            point_set = motor_data[:, 0:3]
            motor_labels = motor_data[:, 6]
            num_eachtype_in_onemotor,_ = np.histogram(motor_labels, bins=7, range=(0,7))
            label_num_eachtype += num_eachtype_in_onemotor
            self.all_points.append(point_set)
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
        n_points = point_set.shape[0]
        choose = np.random.choice(n_points, self.num_points, replace=True)
        chosed_points = point_set[choose, :]
        chosed_labels = labels[choose]        
        chosed_points = pc_normalize(chosed_points)
        return chosed_points, chosed_labels