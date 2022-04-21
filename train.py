#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from unet import unet
from data_loader import TowelDataset, TowelDataset_1
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import weights_init, compute_map, compute_iou, compute_auc, preprocessHeatMap 
# from tensorboardX import SummaryWriter
from PIL import Image
import wandb
import argparse


class BaseTrain:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.init_dirs()
        self.init_datasets()
        self.n_class = self.cfgs["n_class"]
        self.iou_scores = np.zeros((self.cfgs["epochs"], self.n_class))
        self.pixel_scores = np.zeros(self.cfgs["epochs"])
        # self.init_tboard()

    def init_dirs(self):
        runspath = self.cfgs["runspath"] 
        self.run_id = str(max([int(run_id) for run_id in os.listdir(runspath) if run_id.isdecimal()]) + 1)
        self.model_path = os.path.join(runspath, self.run_id)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        with open(os.path.join(self.model_path, 'cfgs.json'), 'w') as f:
            json.dump(self.cfgs, f, sort_keys=True, indent=2)
        self.chkpnts_path = os.path.join(self.model_path, "chkpnts")
        if not os.path.exists(self.chkpnts_path):
            os.makedirs(self.chkpnts_path)
        print(self.model_path)
    
    def init_datasets(self):
        self.datasize = self.cfgs["datasize"] 
        train_data = TowelDataset_1(root_dir=self.cfgs["datapath"], phase='train', num_masks = self.cfgs["n_feature"],use_transform=self.cfgs["transform"], datasize=self.datasize)
        self.train_loader = DataLoader(train_data, batch_size=self.cfgs["batch_size"], shuffle=True, num_workers=16)
        
        #if self.datasize != None and self.datasize <= 8: return
        val_data = TowelDataset_1(root_dir=self.cfgs["datapath"], phase='val',num_masks = self.cfgs["n_feature"], use_transform=False, datasize=self.datasize)
        self.val_loader = DataLoader(val_data, batch_size=self.cfgs["batch_size"], num_workers=16)
    
    def init_model(self):
        self.model = unet(n_classes=self.cfgs["n_feature"], in_channels=1)
        self.use_gpu = torch.cuda.is_available()

    def init_tboard(self):
        self.score_dir = os.path.join(self.model_path, 'scores')
        os.mkdir(self.score_dir)
        self.n_class = self.cfgs["n_class"]
        self.iou_scores = np.zeros((self.cfgs["epochs"], self.n_class))
        self.pixel_scores = np.zeros(self.cfgs["epochs"])

        train_sum_path = os.path.join(self.model_path, "summaries", "train", self.run_id)
        os.makedirs(train_sum_path)
        self.train_writer = SummaryWriter(train_sum_path)

        val_sum_path = os.path.join(self.model_path, "summaries", "val", self.run_id)
        self.val_writer = SummaryWriter(val_sum_path)
        # if self.datasize != None and self.datasize <= 8: return
        # os.makedirs(val_sum_path)

    def loss(self, outputs, labels):
        return self.criterion(outputs[:,:self.n_class,:,:], labels)

    def tboard(self, writer, names, vals, epoch):
        for i in range(len(names)):
            writer.add_scalar(names[i], vals[i], epoch)

    def metrics(self, outputs, labels):
        output = torch.sigmoid(outputs[:,:self.n_class,:,:])
        output = output.data.cpu().numpy()
        pred = output.transpose(0, 2, 3, 1)
        gt = labels.cpu().numpy().transpose(0, 2, 3, 1)

        maps = []
        ious = []
        aucs = []
        for g, p in zip(gt, pred):
            maps.append(compute_map(g, p, self.n_class))
            ious.append(compute_iou(g, p, self.n_class))
            aucs.append(compute_auc(g, p, self.n_class))
        return maps, ious, aucs

class TrainSeg(BaseTrain):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.init_model()

    def init_model(self):
        super().init_model()
        weights_init(self.model)
        if self.use_gpu:
            torch.cuda.device(0)
            self.model = self.model.cuda()
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.).to('cuda'))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfgs["lr"], weight_decay=self.cfgs["w_decay"])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfgs["step_size"], gamma=self.cfgs["gamma"])  # decay LR by a factor of 0.5 every 30 epochs

    def get_batch(self, batch):
        inputs = Variable(batch['X'].cuda()) if self.use_gpu else Variable(batch['X'])
        labels = Variable(batch['Y'].cuda()) if self.use_gpu else Variable(batch['Y'])
        return inputs, labels

    def train(self):
        ts = time.time()
        for epoch in range(self.cfgs["epochs"]):
            self.model.train()
            self.scheduler.step()

            losses = []
            maps, ious, aucs = ([] for i in range(3))
            for iter, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                inputs, labels = self.get_batch(batch)
                outputs = self.model(inputs)

                # Compute losses
                loss = self.loss(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

                if iter % 10 == 0:
                    print("epoch%d, iter%d, loss: %0.5f" % (epoch, iter, loss))
                wandb.log({"train_loss_step":loss})

                batch_maps, batch_ious, batch_aucs = self.metrics(outputs, labels)
                maps += batch_maps
                ious += batch_ious
                aucs += batch_aucs
                del inputs,outputs,batch,labels
                torch.cuda.empty_cache()

            ious = np.nanmean(ious, axis=1)
            pixel_map = np.nanmean(maps)
            mean_auc = np.nanmean(aucs)

            # Write to tensorboard
            names = ['loss', 'MAP', 'meanIOU', 'meanAUC']
            values = [np.nanmean(losses), pixel_map, np.nanmean(ious), mean_auc]
            # self.tboard(self.train_writer, names, values, epoch)
            wandb.log({"train_loss_epoch": np.nanmean(losses),
                        "train_IOU": np.nanmean(ious),
                        "train_AUC": mean_auc})
            print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

            if epoch % 1 == 0:
                #if self.datasize == None or self.datasize > 8:
                self.val(epoch)
                torch.save(self.model.state_dict(), os.path.join(self.chkpnts_path, "%s_epoch%d" % (self.run_id, epoch)))
            
    def val(self, epoch):
        self.model.eval()
        num_batches = len(self.val_loader)
        losses = []
        maps, ious, aucs = ([] for i in range(3))
        for iter, batch in enumerate(self.val_loader):
            inputs, labels = self.get_batch(batch)
            outputs = self.model(inputs)

            with torch.no_grad():
                loss = self.loss(outputs, labels)
                losses.append(loss.item())

                batch_maps, batch_ious, batch_aucs = self.metrics(outputs, labels)
                maps += batch_maps
                ious += batch_ious
                aucs += batch_aucs
            i=0
            if epoch % 20 == 0 and iter == 0:
                hm = torch.sigmoid(outputs[0, :, :, :])
                images = wandb.Image(hm, caption="hm_"+str(epoch))
                wandb.log({"results":images})
                i+=1

            del inputs,outputs,labels,batch
            torch.cuda.empty_cache()

        # Calculate average
        ious = np.array(ious).T  # n_class * val_len
        ious = np.nanmean(ious, axis=1)
        pixel_map = np.nanmean(maps)
        mean_auc = np.nanmean(aucs)
        wandb.log({"val_loss":np.nanmean(losses)})
        wandb.log({"val_pixel_map":pixel_map})
        wandb.log({"val_meanIOU":np.nanmean(ious)})
        wandb.log({"val_meanAUC":mean_auc})
        print("epoch{}, pix_map: {}, meanAUC: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_map, mean_auc, np.nanmean(ious), ious))

if __name__ == '__main__':
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr','--lr', help='learning rate', default = 1e-3) # required=True ,
    parser.add_argument('-wd','--w_decay', help='weight decay (regularization)', default=0) #required=True ,
    parser.add_argument('-m','--momentum', help='momentum (Adam)',  default=0)#required=True ,
    parser.add_argument('-ss','--step_size', help='Description for bar argument',  default=30)#required=True ,
    parser.add_argument('-g','--gamma', help='Description for foo argument', default=0.5)#required=True ,
    parser.add_argument('-bs','--batch_size', help='Description for bar argument', default=8)#required=True ,
    parser.add_argument('-e','--epochs', help='Description for bar argument', default=50)#required=True ,
    parser.add_argument('-dp','--datapath', help='Description for bar argument', default="/home/sashank/deepl_project/data/dataset/test/")#required=True ,
    parser.add_argument('-rp','--runspath', help='Description for bar argument', default="/home/sashank/deepl_project/cloth-segmentation/train_runs")#required=True ,
    parser.add_argument('-t','--transform', help='Description for bar argument', default=True)#required=True ,
    parser.add_argument('-nc','--n_class', help='Description for bar argument', default=2)#required=True ,
    parser.add_argument('-nf','--n_feature', help='Description for bar argument', default=2)#required=True ,
    parser.add_argument('-ds','--datasize', help='Description for bar argument', default="")#required=True ,

    args = vars(parser.parse_args())

    # with open('configs/segmentation.json', 'r') as f:
    #     cfgs = json.loads(f.read())
    # print(json.dumps(cfgs, sort_keys=True, indent=1))
    wandb.init(project="11785project", entity="stirumal", config=args)
    t = TrainSeg(args)
    t.train()
