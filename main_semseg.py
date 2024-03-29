# -*- coding: utf-8 -*-
"""
Created on Fri Jun  12 08:51:52 2022

@author: linux
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from model_cls_seg import DGCNN_cls_semseg
from model import PointNet2_semseg
from torch.utils.data import DataLoader
from data_semseg import MotorDataset, MotorDataset_patch
from util import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes = ['clamping_system', 'cover', 'gear_container', 'charger', 'bottom', 'side_bolt', 'cover_bolt']
labels2categories = {i:cls for i,cls in enumerate(classes)}       # dictionary for labels2categories


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/' + args.model + '/' + args.exp):
        os.makedirs('outputs/' + args.model + '/' + args.exp)
    if not os.path.exists('outputs/' + args.model + '/' + args.exp + '/' + args.change + '/models'):
        os.makedirs('outputs/' + args.model + '/' + args.exp + '/' + args.change + '/models')


def train(args, io):
    NUM_POINT=args.npoints
    print("start loading training data ...")
    TRAIN_DATASET = MotorDataset(split='train', root=args.root, num_points=NUM_POINT, test_area=args.validation_symbol)
    print("start loading test data ...")
    TEST_DATASET = MotorDataset(split='test', root=args.root, num_points=NUM_POINT, test_area=args.validation_symbol)
    train_loader = DataLoader(TRAIN_DATASET, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True, 
                              worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    test_loader = DataLoader(TEST_DATASET, num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    tmp = torch.cuda.max_memory_allocated()
    if args.model == 'dgcnn':
        model = DGCNN_cls_semseg(args).to(device)
    elif args.model == 'pointnet':
        model = PointNet2_semseg(args).to(device)
    else:
        raise Exception("Not implemented")
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    if args.opt == 'sgd':
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    elif args.opt == 'adam':
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'adamw':
        print("Use AdamW")
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.1, args.epochs)
    
    criterion = cal_loss
    NUM_CLASS = 7
    best_iou = 0
    best_bolts_iou = 0
    
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        num_batches=len(train_loader)
        total_correct=0
        total_seen=0
        loss_sum=0
        model=model.train()
        args.training=True
        total_correct_class__ = [0 for _ in range(NUM_CLASS)]
        total_iou_deno_class__ = [0 for _ in range(NUM_CLASS)]
        
        for i, (points, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            points, target = points.to(device), target.to(device)
            # points = normalize_data(points)
            points = points.permute(0, 2, 1)                            #(batch_size, features, num_points)
            batch_size = points.size()[0]
            opt.zero_grad()
            seg_pred, cls_pred = model(points.float())        #(batch_size, class_categories, num_points)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()                      #(batch_size,num_points, class_categories)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()              #array(batch_size*num_points)
            loss = criterion(seg_pred.view(-1, NUM_CLASS), target.view(-1,1).squeeze())     #a scalar
            loss.backward()
            opt.step()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASS)                    # (batch_size*num_points , num_class)
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()                     # array(batch_size*num_points)
            correct = np.sum(pred_choice == batch_label)                            # when a np arraies have same shape, a==b means in conrrespondending position it equals to one,when they are identical
            total_correct += correct
            total_seen += (batch_size * NUM_POINT)
            loss_sum += loss
            for l in range(NUM_CLASS):
                total_correct_class__[l] += np.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class__[l] += np.sum(((pred_choice == l) | (batch_label == l)))
        mIoU__ = np.mean(np.array(total_correct_class__) / (np.array(total_iou_deno_class__, dtype=np.float64) + 1e-6))
        cb_IoU = total_correct_class__[6]/float(total_iou_deno_class__[6])
        Bolt_IoU = (total_correct_class__[6]+total_correct_class__[5]) / (float(total_iou_deno_class__[6])+float(total_iou_deno_class__[5]))

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train mIoU:%.5f, cb_IoU:%.5f, Bolt_IoU:%.5f' % (epoch,
            (loss_sum / num_batches),(total_correct / float(total_seen)), mIoU__, cb_IoU, Bolt_IoU)
        io.cprint(outstr)
        writer.add_scalar('Loss/Train mean Loss', (loss_sum / num_batches), epoch)
        writer.add_scalar('Accuracy/Train accuracy', (total_correct / float(total_seen)), epoch)
        writer.add_scalar('mIoU/Train mean IoU', mIoU__, epoch)
        writer.add_scalar('IoU of cover_bolt/train', cb_IoU, epoch)
        writer.add_scalar('IoU of bolt/train', Bolt_IoU, epoch)
        ####################
        # Validation
        ####################
        with torch.no_grad():         
            num_batches = len(test_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASS)
            total_seen_class = [0 for _ in range(NUM_CLASS)]
            total_correct_class = [0 for _ in range(NUM_CLASS)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASS)]
            noBG_seen_class = [0 for _ in range(NUM_CLASS-1)]
            noBG_correct_class = [0 for _ in range(NUM_CLASS-1)]
            noBG_iou_deno_class = [0 for _ in range(NUM_CLASS-1)]
            model = model.eval()
            args.training = False

            for i, (points, seg) in tqdm(enumerate(test_loader),total=len(test_loader),smoothing=0.9):
                points, seg = points.to(device), seg.to(device)
                # points = normalize_data(points)
                points = points.permute(0, 2, 1)
                batch_size = points.size()[0]
                seg_pred, cls_pred = model(points.float())
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()   # array(batch_size*num_points)
                loss = criterion(seg_pred.view(-1, NUM_CLASS), seg.view(-1,1).squeeze())
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASS)   # (batch_size*num_points, num_class)
                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (batch_size * NUM_POINT)
                loss_sum += loss
                tmp, _ = np.histogram(batch_label, range(NUM_CLASS + 1))
                labelweights += tmp
                for l in range(NUM_CLASS):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum((pred_choice == l) | (batch_label == l))
              
                ####### calculate without Background ##############
                for l in range(1, NUM_CLASS):
                    noBG_seen_class[l-1] += np.sum((batch_label == l))
                    noBG_correct_class[l-1] += np.sum((pred_choice == l) & (batch_label == l))
                    noBG_iou_deno_class[l-1] += np.sum((pred_choice == l) | (batch_label == l))
            
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
            cb_IoU_wB = total_correct_class[6]/float(total_iou_deno_class[6])
            Bolt_IoU_wB = (total_correct_class[6]+total_correct_class[5]) / (float(total_iou_deno_class[6])+float(total_iou_deno_class[5]))

            outstr = 'Validation with backgroud----epoch: %d,  eval mean loss %.6f,  eval mIoU %.6f, eval_cb_IoU:%5.f, eval_bolt_Iou:%.5f, eval point acc %.6f, eval point avg class IoU %.6f' % (epoch,(loss_sum / num_batches),mIoU,
                                                        cb_IoU_wB, Bolt_IoU_wB, (total_correct / float(total_seen)),(np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
            io.cprint(outstr)
            noBG_mIoU = np.mean(np.array(noBG_correct_class) / (np.array(noBG_iou_deno_class, dtype=np.float64) + 1e-6))
            outstr_without_background='Validation without backgroud----epoch: %d, mIoU %.6f,  eval point accuracy: %.6f, eval point avg class acc: %.6f' % (epoch,noBG_mIoU,
                                                        (sum(noBG_correct_class) / float(sum(noBG_seen_class))),(np.mean(np.array(noBG_correct_class) / (np.array(noBG_seen_class, dtype=np.float64) + 1e-6))))
            io.cprint(outstr_without_background)

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASS):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    labels2categories[l] + ' ' * (14 - len(labels2categories[l])), labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))
            io.cprint(iou_per_class_str)

            if mIoU >= best_iou:
                best_iou = mIoU
                savepath = str(BASE_DIR)+'/outputs/'+args.model +'/'+ args.exp +'/' + args.change + '/models/best_m.pth'
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }
                io.cprint('Saving best model at %s' % savepath)
                torch.save(state, savepath)
            io.cprint('Best mIoU: %f' % best_iou)
            cur_bolts_iou=(total_correct_class[5] + total_correct_class[6]) / (float(total_iou_deno_class[5]) + float(total_iou_deno_class[6]))
            
            if cur_bolts_iou >= best_bolts_iou:
                best_bolts_iou=cur_bolts_iou
                savepath = str(BASE_DIR)+'/outputs/'+args.model +'/'+ args.exp +'/' + args.change + '/models/best.pth'
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }
                io.cprint('Saving best model at %s' % savepath)
                torch.save(state, savepath)
            io.cprint('Best IoU of bolts: %f' % best_bolts_iou)
            io.cprint('\n\n')
        writer.add_scalar('learning rate', opt.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/Validation mean loss', loss_sum / num_batches, epoch)
        writer.add_scalar('Accuracy/Validation accuracy', total_correct / float(total_seen), epoch)
        writer.add_scalar('mIoU/Validation mean MoU', mIoU, epoch)
        writer.add_scalar('IoU of bolt/Validation', cur_bolts_iou, epoch)
        writer.add_scalar('IoU of cover_bolt/Validation', Bolt_IoU_wB, epoch)
    io.close()


def test(args, io):
    NUM_POINT = args.npoints
    print("start loading test data ...")
    TEST_DATASET = MotorDataset_patch(split='test', root=args.root, num_points=NUM_POINT, test_area=args.validation_symbol)
    test_loader = DataLoader(TEST_DATASET, num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'dgcnn':
        model = DGCNN_cls_semseg(args).to(device)
    else:
        raise Exception("Not implemented !")

    model = nn.DataParallel(model)
    print("Let's test and use ", torch.cuda.device_count(), " GPUs!")

    try:
        if args.eval:
            checkpoint = torch.load(str(args.model_path))
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print("No pretrained model exists!")
        exit(-1)

    criterion = cal_loss
    NUM_CLASS = 7

    ##################### Test ####################
    with torch.no_grad():
        num_batches = len(test_loader)
        total_correct = 0
        total_seen = 0
        labelweights = np.zeros(NUM_CLASS)
        total_seen_class = [0 for _ in range(NUM_CLASS)]
        total_correct_class = [0 for _ in range(NUM_CLASS)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASS)]
        model = model.eval()

        for i, (points, seg) in tqdm(enumerate(test_loader), total = len(test_loader), smoothing=0.9):
            points, seg = points.to(device), seg.to(device)
            points = points.permute(0, 2, 1)
            batch_size = points.size()[0]
            seg_pred, cls_pred = model(points.float())
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()   # array(batch_size*num_points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASS)   # (batch_size*num_points, num_class)
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (batch_size * NUM_POINT)
            tmp, _ = np.histogram(batch_label, range(NUM_CLASS + 1))
            labelweights += tmp
            for l in range(NUM_CLASS):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum((pred_choice == l) | (batch_label == l))
        
        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
        cb_IoU_wB = total_correct_class[6]/float(total_iou_deno_class[6])
        screw_IoU_wB = (total_correct_class[6]+total_correct_class[5]) / (float(total_iou_deno_class[6])+float(total_iou_deno_class[5]))
        
        outstr = 'Test with backgound: ---mIoU: %.6f, ---screw_IoU: %.6f, ---cover_screw_IoU: %.6f' % (
            mIoU, screw_IoU_wB, cb_IoU_wB)
        io.cprint(outstr)

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASS):
            iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                labels2categories[l] + ' ' * (14 - len(labels2categories[l])), labelweights[l],
                total_correct_class[l] / float(total_iou_deno_class[l]))
        io.cprint(iou_per_class_str)
        io.cprint('\n\n')



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'pointnet'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--root', type=str, default='/home/bi/study/thesis/data/test', 
                        help='file need to be tested')
    parser.add_argument('--exp', type=str, default='training_125', metavar='N',
                        help='experiment version to record result')
    parser.add_argument('--change', type=str, default='hh', metavar='N',
                        help='experiment version to record result')
    parser.add_argument('--finetune', type=bool, default=False, metavar='N',
                        help='if we finetune the model')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--training', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--opt', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'],
                        help='optimizer to choose, [SGD, Adam, AdamW]')
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
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=32, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--npoints', type=int, default=2048, 
                        help='Point Number [default: 2048]')
    parser.add_argument('--validation_symbol', type=str, default='Validation', 
                        help='Which datablocks to use for validation')
    parser.add_argument('--test_symbol', type=str, default='Test', 
                        help='Which datablocks to use for test')
    parser.add_argument('--hidden_size', type=int, default=512, metavar='hidden_size',
                        help='number of hidden_size for self_attention ')
    parser.add_argument('--model_path', type=str, default='NN',
                        help='path of pretrained model')
    args = parser.parse_args()
    
    _init_()
    
    writer = SummaryWriter('outputs/' + args.model + '/' + args.exp + '/' + args.change)
    
    io = IOStream('outputs/' + args.model + '/' + args.exp + '/' + args.change + '/result.log')
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
    else:
        test(args, io)
        