# -*- coding: utf-8 -*-
"""
Created on Tue May 24 04:09:50 2022

@author: linux
"""

import random
import os
import shutil




def random_ls(num_motors):
    num_train = int(0.8 * num_motors)
    # num_val = int(0.2 * num_motors)
    ls = []
    val_ls = []
    for i in range(num_motors):
        ls.append(i + 1)
    train_ls = random.sample(ls, num_train)
    for n in ls:
        if n not in train_ls:
            val_ls.append(n)
    return train_ls, val_ls
            
    

base_dir = 'E:\\test'  # dir of the raw data
dst_dir = 'E:\\dataset'    # dir of the training data
ls_type = os.listdir(base_dir)
if 'motor_parameter.csv' in ls_type:
    ls_type.remove('motor_parameter.csv')
ls_type.sort()

for types in ls_type:
    type_dir = base_dir + '\\' + types
    ls_motor = os.listdir(type_dir)
    if 'camera_motor_setting.csv' in ls_motor:
        ls_motor.remove('camera_motor_setting.csv')
    if 'motor_3D_bounding_box.csv' in ls_motor:
        ls_motor.remove('motor_3D_bounding_box.csv')
    ls_motor.sort()
    train_ls, val_ls = random_ls(len(ls_motor))
    for motor in ls_motor:       
        motor_dir = type_dir + '\\' + motor
        k = motor.split('_')
        if int(k[1]) in train_ls:
            file_name = types + '_' + k[1] + '_cuboid.npy'
            src = motor_dir + '\\' + file_name
            dst = dst_dir + '\\Training_' + file_name
            shutil.copyfile(src, dst)
        if int(k[1]) in val_ls:
            file_name = types + '_' + k[1] + '_cuboid.npy'
            src = motor_dir + '\\' + file_name
            dst = dst_dir + '\\Validation_' + file_name
            shutil.copyfile(src, dst)
                
        
            
        
    
    