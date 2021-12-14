from __future__ import print_function
import os
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from sklearn.model_selection import train_test_split
import numpy as np

from src.dataset import get_dataset
import src.utils as utils

parser = argparse.ArgumentParser(description="FIXBI EVALUATION")
parser.add_argument('-db_path', help='gpu number', type=str, default='datasets')
parser.add_argument('-save_path', help='save path', type=str, default='Logs/original_algo')
parser.add_argument('-source', help='source', type=str, default='dslr')
parser.add_argument('-target', help='target', type=str, default='webcam')
parser.add_argument('-workers', default=4, type=int, help='dataloader workers')
parser.add_argument('-gpu', help='gpu number', type=str, default='0,1')
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-met_file', default="met_file_amazon_dslr.txt", type=str, help='Metric file')

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(args)

    num_classes, resnet_type = utils.get_data_info()
    src_trainset, src_testset = get_dataset(args.source, path=args.db_path)
    tgt_trainset, tgt_testset = get_dataset(args.target, path=args.db_path)

    src_test_loader = torchdata.DataLoader(src_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
    tgt_test_loader = torchdata.DataLoader(tgt_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    net_sd, head_sd, classifier_sd = utils.get_net_info(num_classes)
    models_sd = [net_sd, head_sd, classifier_sd]
    
    net_td, head_td, classifier_td = utils.get_net_info(num_classes)
    models_td = [net_td, head_td, classifier_td]

    net_sd, head_sd, classifier_sd = utils.load_net(args, 'sdm', net_sd, head_sd, classifier_sd)
    net_td, head_td, classifier_td = utils.load_net(args, 'tdm', net_td, head_td, classifier_td)
                
    utils.evaluate(nn.Sequential(*models_sd), src_test_loader, args.met_file)
    utils.evaluate(nn.Sequential(*models_sd), tgt_test_loader, args.met_file)
    utils.evaluate(nn.Sequential(*models_td), tgt_test_loader, args.met_file)    
    utils.final_eval(nn.Sequential(*models_sd), nn.Sequential(*models_td), tgt_test_loader, args.met_file)

    utils.evaluate_class_wise(nn.Sequential(*models_sd), src_test_loader, args.met_file)
    utils.evaluate_class_wise(nn.Sequential(*models_sd), tgt_test_loader, args.met_file)
    utils.evaluate_class_wise(nn.Sequential(*models_td), tgt_test_loader, args.met_file)    
    # utils.final_eval_class_wise(nn.Sequential(*models_sd), nn.Sequential(*models_td), tgt_test_loader, args.met_file)

main()