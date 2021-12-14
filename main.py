from __future__ import print_function
import os
import copy
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from sklearn.model_selection import train_test_split
import numpy as np

from trainer.fixbi_trainer import train_fixbi
from src.dataset import get_dataset
import src.utils as utils

parser = argparse.ArgumentParser(description="FIXBI EXPERIMENTS")
parser.add_argument('-db_path', help='gpu number', type=str, default='datasets')
parser.add_argument('-baseline_path', help='baseline path', type=str, default='AD_Baseline')
parser.add_argument('-save_path', help='save path', type=str, default='Logs/save_test/lr_ratio')
parser.add_argument('-source', help='source', type=str, default='webcam')
parser.add_argument('-target', help='target', type=str, default='amazon')
parser.add_argument('-workers', default=4, type=int, help='dataloader workers')
parser.add_argument('-gpu', help='gpu number', type=str, default='0,1')
parser.add_argument('-pretrain_epochs', default=10, type=int)
parser.add_argument('-epochs', default=200, type=int)
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-th', default=2.0, type=float, help='Threshold')
parser.add_argument('-bim_start', default=100, type=int, help='Bidirectional Matching')
parser.add_argument('-sp_start', default=25, type=int, help='Self-Penalization')
parser.add_argument('-cr_start', default=100, type=int, help='Consistency Regularization')
parser.add_argument('-lam_sd', default=0.7, type=float, help='Source Dominant Mixup ratio')
parser.add_argument('-lam_td', default=0.3, type=float, help='Target Dominant Mixup ratio')
parser.add_argument('-met_file', default="met_file_amazon_dslr.txt", type=str, help='Metric file')

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.autograd.set_detect_anomaly(True)
    print("Use GPU(s): {} for training".format(args.gpu))
    print(args)

    num_classes, resnet_type = utils.get_data_info()
    src_trainset, src_testset = get_dataset(args.source, path=args.db_path)
    tgt_trainset, tgt_testset = get_dataset(args.target, path=args.db_path)

    targets = tgt_trainset.targets

    src_train_loader1 = torchdata.DataLoader(src_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_train_loader1 = torchdata.DataLoader(tgt_trainset, batch_size=args.batch_size,shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    src_train_loader2 = torchdata.DataLoader(src_trainset, batch_size=args.batch_size//2, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_train_loader2 = torchdata.DataLoader(tgt_trainset, batch_size=args.batch_size//2,shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_test_loader = torchdata.DataLoader(tgt_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    lr, l2_decay, momentum, nesterov = utils.get_train_info()
    lambda1 = lambda epoch: 1/((1 + 10*epoch/args.epochs)**0.75)    

    for lr_ratio in [0.1]:

        net_sd, head_sd, classifier_sd = utils.get_net_info(num_classes)
        net_td, head_td, classifier_td = utils.get_net_info(num_classes)
        net_sd, head_sd, classifier_sd = utils.load_net(args, 'sdm_epoch100', net_sd, head_sd, classifier_sd)
        net_td, head_td, classifier_td = utils.load_net(args, 'tdm_epoch100', net_td, head_td, classifier_td)
        
        learnable_params_sd = list(net_sd.parameters()) + list(head_sd.parameters()) + list(classifier_sd.parameters())
        learnable_params_td = list(net_td.parameters()) + list(head_td.parameters()) + list(classifier_td.parameters())
        
        models_sd = [net_sd, head_sd, classifier_sd]
        models_td = [net_td, head_td, classifier_td]
        
        sp_param_sd = nn.Parameter(torch.tensor(5.0).cuda(), requires_grad=True)
        sp_param_td = nn.Parameter(torch.tensor(5.0).cuda(), requires_grad=True)
        
        optimizer_sd = optim.SGD(learnable_params_sd, lr=lr, momentum=momentum, weight_decay=l2_decay, nesterov=nesterov)
        optimizer_td = optim.SGD(learnable_params_td, lr=lr, momentum=momentum, weight_decay=l2_decay, nesterov=nesterov)
        optimizer_sd.add_param_group({"params": [sp_param_sd], "lr": lr})
        optimizer_td.add_param_group({"params": [sp_param_td], "lr": lr})

        scheduler_sd = torch.optim.lr_scheduler.LambdaLR(optimizer_sd, lr_lambda=lambda1)
        scheduler_td = torch.optim.lr_scheduler.LambdaLR(optimizer_td, lr_lambda=lambda1)
        
        ce = nn.CrossEntropyLoss().cuda()
        mse = nn.MSELoss().cuda()
        
        loaders = [src_train_loader1, tgt_train_loader1, src_train_loader2, tgt_train_loader2]
        optimizers = [optimizer_sd, optimizer_td, lr_ratio]
        schedulers = [scheduler_sd,scheduler_td]
        sp_params = [sp_param_sd, sp_param_td]  
        losses = [ce, mse]

        p1,p2,p3,p4 = [],[],[],[]
        f1,f2 = [],[]

        p1.append(utils.evaluate(nn.Sequential(*models_sd), src_train_loader1, args.met_file))
        p2.append(utils.evaluate(nn.Sequential(*models_td), src_train_loader1, args.met_file))
        f1.append(utils.final_eval(nn.Sequential(*models_sd), nn.Sequential(*models_td), src_train_loader1, args.met_file))

        p3.append(utils.evaluate(nn.Sequential(*models_sd), tgt_test_loader, args.met_file))
        p4.append(utils.evaluate(nn.Sequential(*models_td), tgt_test_loader, args.met_file))
        p2.append(utils.final_eval(nn.Sequential(*models_sd), nn.Sequential(*models_td), tgt_test_loader, args.met_file))

        for epoch in range(100,120):
            train_fixbi(args, loaders, optimizers, schedulers, models_sd, models_td, sp_params, losses, epoch, args.met_file)
            
            p1.append(utils.evaluate(nn.Sequential(*models_sd), src_train_loader1, args.met_file))
            p2.append(utils.evaluate(nn.Sequential(*models_td), src_train_loader1, args.met_file))
            f1.append(utils.final_eval(nn.Sequential(*models_sd), nn.Sequential(*models_td), src_train_loader1, args.met_file))

            p3.append(utils.evaluate(nn.Sequential(*models_sd), tgt_test_loader, args.met_file))
            p4.append(utils.evaluate(nn.Sequential(*models_td), tgt_test_loader, args.met_file))
            f2.append(utils.final_eval(nn.Sequential(*models_sd), nn.Sequential(*models_td), tgt_test_loader, args.met_file))

            if epoch < 100:
                utils.save_net(args, models_sd, 'sdm_epoch100')
                utils.save_net(args, models_td, 'tdm_epoch100')
            else:
                utils.save_net(args, models_sd, 'sdm_epoch120_'+str(lr_ratio))
                utils.save_net(args, models_td, 'tdm_epoch120_'+str(lr_ratio))

        print(f1)
        print(f2)
        print(p1)
        print(p2)
        print(p3)
        print(p4)

        x = range(100,121)
        plt.plot(x,p1,color='red',label='SDM Accuracy on Source Dataset')
        plt.plot(x,p2,color='blue',label='TDM Accuracy on Source Dataset')
        plt.plot(x,p3,color='cyan',label='SDM Accuracy on Target Dataset')
        plt.plot(x,p4,color='green',label='TDM Accuracy on Target Dataset')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(s+'_'+t+'_lr_ratio_'+lr_ratio+'training_plot.png')

        plt.plot(x,f1,color='red',label='Final Accuracy on Source Dataset')
        plt.plot(x,f2,color='green',label='Final Accuracy on Target Dataset')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(s+'_'+t+'final_lr_ratio_'+lr_ratio+'training_plot.png')

main()