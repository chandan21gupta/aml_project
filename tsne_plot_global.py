from __future__ import print_function
import os
import copy
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from sklearn.model_selection import train_test_split
import numpy as np

from trainer.fixbi_trainer import train_fixbi, pretrain
from src.dataset import get_dataset
import src.utils as utils
from random import randint, seed

parser = argparse.ArgumentParser(description="FIXBI EXPERIMENTS")
parser.add_argument('-db_path', help='gpu number', type=str, default='datasets')
parser.add_argument('-save_path', help='save path', type=str, default='Logs/save_test/lr_ratio')
parser.add_argument('-source', help='source', type=str, default='webcam')
parser.add_argument('-target', help='target', type=str, default='amazon')
parser.add_argument('-workers', default=4, type=int, help='dataloader workers')
parser.add_argument('-gpu', help='gpu number', type=str, default='0,1')
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-met_file', default="met_file_dslr_webcam.txt", type=str, help='Metric file')

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Use GPU(s): {} for training".format(args.gpu))

    color = []
    seed(0)
    num_classes, resnet_type = utils.get_data_info()
    for i in range(num_classes):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    color = np.array(color)

    net_sd, head_sd, classifier_sd = utils.get_net_info(num_classes)
    net_td, head_td, classifier_td = utils.get_net_info(num_classes)

    for k in [1,0.01]:#,("amazon","webcam"),("webcam","dslr"),("dslr","webcam")]:
        # args.source = src
        # args.target = tgt
        print(args.source,args.target)

        src_trainset, src_testset = get_dataset(args.source, path=args.db_path)
        tgt_trainset, tgt_testset = get_dataset(args.target, path=args.db_path)

        src_train_loader1 = torchdata.DataLoader(src_trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
        tgt_train_loader1 = torchdata.DataLoader(tgt_trainset, batch_size=args.batch_size,shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

        net_sd, head_sd, classifier_sd = utils.load_net(args, 'sdm_epoch120_'+str(k), net_sd, head_sd, classifier_sd)
        net_td, head_td, classifier_td = utils.load_net(args, 'tdm_epoch120_'+str(k), net_td, head_td, classifier_td)
        models_sd = [net_sd, head_sd, classifier_sd]
        models_td = [net_td, head_td, classifier_td]

        for l,loader in [('tgt',tgt_train_loader1),]: #('src',src_train_loader1),

            #for m,modelx in [('sdm',models_sd),('tdm',models_td),]: #
            model = nn.Sequential(*models_sd[:-1])
            utils.set_model_mode('eval', [*models_sd])
            X1, Y = torch.empty(size=[0,256]), torch.empty(size=[0]) 
            for step, (x,y) in enumerate(loader):
                try:
                    x = x.cuda(non_blocking=True)
                    x = model(x).cpu()
                    X1 = torch.cat((X1,x.detach()),0)
                    Y = torch.cat((Y,y.detach()),0)
                except:
                    print(X1.size(),x.size())

            model = nn.Sequential(*models_td[:-1])
            utils.set_model_mode('eval', [*models_td])
            X2 = torch.empty(size=[0,256])
            for step, (x,y) in enumerate(loader):
                try:
                    x = x.cuda(non_blocking=True)
                    x = model(x).cpu()
                    X2 = torch.cat((X2,x.detach()),0)
                except:
                    print(X2.size(),x.size())

            X1 = X1 + X2

            # tsne_proj = PCA(n_components=0.9).fit_transform(X.numpy())
            # print(np.shape(tsne_proj))
            print("src:",args.source,"tgt:",args.target,"lr_ratio:",k)
            utils.final_eval(nn.Sequential(*models_sd), nn.Sequential(*models_td), loader, args.met_file)
            tsne_proj = TSNE(2, perplexity=30, learning_rate=200, verbose=1).fit_transform(X1.numpy())
            Y = Y.numpy().astype(int)
            print(len(Y))
            df = pd.DataFrame()
            df["comp-1"] = tsne_proj[:,0]
            df["comp-2"] = tsne_proj[:,1]
            df.plot.scatter(x="comp-1", y="comp-2", s=20, c=color[Y])
            plt.savefig(str(k)+' '+'final_'+str(l)+'.png')
            plt.show()

main()