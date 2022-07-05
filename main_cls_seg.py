# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:53:33 2022

@author: linux
"""

import os
import argparse
import torch
import torch.nn as nn
import time
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data_cls import MotorDataset
from model_cls_semseg import DGCNN_cls_semseg
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes = ['clamping_system', 'cover', 'gear_container', 'charger', 'bottom', 'side_bolt', 'cover_bolt']
labels2categories = {i:cls for i,cls in enumerate(classes)}       #dictionary for labels2categories


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/' + args.model + '/' + args.exp_name):
        os.makedirs('outputs/' + args.model + '/' + args.exp_name)
    if not os.path.exists('outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/models'):
        os.makedirs('outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/models')


def train(args, io):
    train_loader = DataLoader(MotorDataset(root=args.root, split='train', 
                                           num_points=args.num_points, 
                                           test_area=args.validation_symbol),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    test_loader = DataLoader(MotorDataset(root=args.root, split='test', 
                                          num_points=args.num_points,
                                          test_area=args.validation_symbol),
                             num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=False)
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    if args.model == 'dgcnn':
        model = DGCNN_cls_semseg(args).to(device)
    else:
        raise Exception('Not implemneted')
    print(str(model))
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs")
    
    if args.opt == 'sgd':
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, 
                        weight_decay=1e-4)
    elif args.opt == 'adam':
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'adamw':
        print("Use AdamW")
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=60, gamma=0.2)
    
    print("Starting from scratch!")
    
    criterion = cal_loss
    # loss_cluster = nn.MSELoss()
    num_cls = 7
    best_iou = 0
    best_bolts_iou = 0
    best_val_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        num_batches = len(train_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        model = model.train()
        args.training = True
        total_correct_class__ = [0 for _ in range(num_cls)]
        total_iou_deno_class__ = [0 for _ in range(num_cls)]
        train_loss = 0.0
        count = 0
        cls_pred = []
        cls_true = []
        for i, (data, seg, types) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            data, seg, types = data.to(device), seg.to(device), types.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits, seg_pred = model(data.float())
            loss_cls = criterion(logits, types)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()
            loss_seg = criterion(seg_pred.view(-1, num_cls), seg.view(-1, 1).squeeze())
            loss = 0.5*loss_cls + loss_seg
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            cls_true.append(types.cpu().numpy())
            cls_pred.append(preds.detach().cpu().numpy())
            seg_pred = seg_pred.contiguous().view(-1, num_cls)
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (batch_size * args.num_points)
            loss_sum += loss
            for l in range(num_cls):
                total_correct_class__[l] += np.sum((pred_choice == l) & (batch_label == l))    # Intersection
                total_iou_deno_class__[l] += np.sum((pred_choice == l) | (batch_label == l))   # Union
        mIoU__ = np.mean(np.array(total_correct_class__) / (np.array(total_iou_deno_class__, dtype=np.float64) + 1e-6))
        
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        
        cls_true = np.concatenate(cls_true)
        cls_pred = np.concatenate(cls_pred)
        outstr_cls = 'Train %d, loss: %.6f, cls train acc: %.6f, cls train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     cls_true, cls_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     cls_true, cls_pred))
        io.cprint(outstr_cls)
        writer.add_scalar('learning rate/lr', opt.param_groups[0]['lr'], epoch)
        # writer.add_scalar('Loss/train loss', train_loss*1.0/count, epoch)
        writer.add_scalar('Accuracy/train cls acc', metrics.accuracy_score(cls_true, cls_pred), epoch)
        writer.add_scalar('Average Accuracy/train avg acc', metrics.balanced_accuracy_score(cls_true, cls_pred), epoch)
        outstr_sem_seg = 'Train %d, seg loss: %.6f, seg train acc: %.6f ' % (epoch, 
                                                              loss_sum / num_batches,
                                                              total_correct / float(total_seen))
        io.cprint(outstr_sem_seg)
        writer.add_scalar('Loss/Train mean Loss', loss_sum / num_batches, epoch)
        writer.add_scalar('Accuracy/Train seg accuracy', (total_correct / float(total_seen)), epoch)
        writer.add_scalar('mIoU/Train mean IoU', mIoU__, epoch)
        writer.add_scalar('IoU of cover_bolt/train', total_correct_class__[6]/float(total_iou_deno_class__[6]), epoch)
        writer.add_scalar('IoU of bolt/train', (total_correct_class__[6] + total_correct_class__[5]) / (float(total_iou_deno_class__[6]) + float(total_iou_deno_class__[5])), epoch)
        
        ####################
        # Validation
        ####################
        with torch.no_grad():
            num_batches = len(test_loader)
            val_loss = 0.0
            count = 0.0
            val_pred = []
            val_true = []
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_seen_class = [0 for _ in range(num_cls)]
            total_correct_class = [0 for _ in range(num_cls)]
            total_iou_deno_class = [0 for _ in range(num_cls)]
            noBG_seen_class = [0 for _ in range(num_cls - 1)]
            noBG_correct_class = [0 for _ in range(num_cls - 1)]
            noBG_iou_deno_class = [0 for _ in range(num_cls - 1)]
            model = model.eval()
            
            for i, (data, seg, types) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                data, seg, types = data.to(device), seg.to(device), types.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits, seg_pred = model(data.float())
                loss_cls = criterion(logits, types)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()
                loss_seg = criterion(seg_pred.view(-1, num_cls), seg.view(-1, 1).squeeze())
                loss = loss_cls + loss_seg
                preds = logits.max(dim=1)[1]
                count += batch_size
                val_loss += loss.item() * batch_size
                val_true.append(types.cpu().numpy())
                val_pred.append(preds.detach().cpu().numpy())
                ### segmentation
                seg_pred = seg_pred.contiguous().view(-1, num_cls)  # (batch_size*num_point, num_class)
                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()   # array(batch_size*num_point)
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (batch_size * args.num_points)
                loss_sum += loss
                for l in range(num_cls):
                    total_seen_class[l] += np.sum(batch_label == l)
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))     ### Intersection
                    total_iou_deno_class[l] += np.sum((pred_choice == l) | (batch_label == l))   ### Union
                
                ####### calculate without Background ##############
                for l in range(1, num_cls):
                    noBG_seen_class[l-1] += np.sum(batch_label == l)
                    noBG_correct_class[l-1] += np.sum((pred_choice == l) & (batch_label == l))     ### Intersection
                    noBG_iou_deno_class[l-1] += np.sum((pred_choice == l) | (batch_label == l))     ### Union
            
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
            
            ###### Classification Results (Validation) #######
            val_true = np.concatenate(val_true)
            val_pred = np.concatenate(val_pred)
            val_acc = metrics.accuracy_score(val_true, val_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(val_true, val_pred)
            outstr_cls_val = 'Test %d, loss: %.6f, test cls acc: %.6f, test avg cls acc: %.6f' % (epoch,
                                                                                  val_loss*1.0/count,
                                                                                  val_acc,
                                                                                  avg_per_class_acc)
            io.cprint(outstr_cls_val)
            writer.add_scalar('Loss/val loss', val_loss*1.0/count, epoch)
            writer.add_scalar('Accuracy/val cls acc', val_acc, epoch)
            writer.add_scalar('Average Accuracy/val avg cls acc', avg_per_class_acc, epoch)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'outputs/%s/%s/%s/models/best_cls_model.t7' % (args.model, args.exp_name, args.change))
            
            ###### Segmentation Results (Validation) ######
            outstr = 'Validation with backgroud----epoch: %d,  eval mean loss %.6f,  eval mIoU %.6f,  eval point acc %.6f, eval point avg class IoU %.6f' % (epoch, loss_sum / num_batches, mIoU,
                                                        total_correct / float(total_seen), np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6)))     
            io.cprint(outstr)
            noBG_mIoU = np.mean(np.array(noBG_correct_class) / (np.array(noBG_iou_deno_class, dtype=np.float64) + 1e-6))
            outstr_without_background='Validation without backgroud----epoch: %d, mIoU %.6f,  eval point accuracy: %.6f, eval point avg class acc: %.6f' % (epoch,noBG_mIoU,
                                                        (sum(noBG_correct_class) / float(sum(noBG_seen_class))),(np.mean(np.array(noBG_correct_class) / (np.array(noBG_seen_class, dtype=np.float64) + 1e-6))))
            io.cprint(outstr_without_background)
            
            iou_per_class_str = '------- IoU --------\n'
            for i in range(num_cls):
                iou_per_class_str += 'class %s IoU: %.3f \n' % (
                    labels2categories[i] + ' ' * (20 - len(labels2categories[i])),
                    total_correct_class[i] / float(total_iou_deno_class[i]))
            io.cprint(iou_per_class_str)
            
            if mIoU >= best_iou:
                best_iou = mIoU
                save_path = str(BASE_DIR) + '/outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/models/best_m.pth'
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict()}
                io.cprint('Save best model at %s' % save_path)
                torch.save(state, save_path)
            io.cprint('Best mIoU: %f' % best_iou)
            cur_bolts_iou = (total_correct_class[5] + total_correct_class[6]) / (float(total_iou_deno_class[5]) + float(total_iou_deno_class[6]))
            if cur_bolts_iou >= best_bolts_iou:
                best_bolts_iou=cur_bolts_iou
                save_path = str(BASE_DIR) + '/outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/models/best.pth'
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }
                io.cprint('Saving best model at %s' % save_path)
                torch.save(state, save_path)
            io.cprint('Best IoU of bolts: %f' % best_bolts_iou)
            io.cprint('\n\n')
        # writer.add_scalar('learning rate', opt.param_groups[0]['lr'], epoch)
        # writer.add_scalar('Loss/Validation mean loss', loss_sum / num_batches, epoch)
        writer.add_scalar('Accuracy/Validation accuracy', total_correct / float(total_seen), epoch)
        writer.add_scalar('mIoU/Validation mean MoU', mIoU, epoch)
        writer.add_scalar('IoU of bolt/Validation', (total_correct_class[5] + total_correct_class[6]) / (float(total_iou_deno_class[5]) + float(total_iou_deno_class[6])), epoch)
        writer.add_scalar('IoU of cover_bolt/Validation', total_correct_class[6] / float(total_iou_deno_class[6]), epoch)

    io.close()
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Classification')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--change', type=str, default='hh', metavar='N',
                        help='explict parameters in experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'pct'],
                        help='Model to use, [dgcnn, pct]')
    parser.add_argument('--root', type=str, metavar='N',default='E:\\dataset',
                        help='folder of dataset')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--opt', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help='optimizer to use, [SGD, Adam, AdamW]')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--validation_symbol', type=str, default='Validation', 
                        help='Which datablocks to use for validation')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    
    _init_()

    writer = SummaryWriter('outputs/' + args.model + '/' + args.exp_name + '/' + args.change)
    
    io = IOStream('outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/result.log')
    io.cprint(str(args))
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    
    if not args.eval:
        train(args, io)  