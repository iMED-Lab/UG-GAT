#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :test.py
@Description :
@Time        :2021/04/12 09:17:12
@Author      :Jinkui Hao
@Version     :1.0
'''


import os
import sys
from torch.utils.data import DataLoader
from torch import optim, nn
import torch
from utils.Visualizer import Visualizer
from  BayesianCNN.models import ResNet
import numpy as np
import csv
from evaluation.matrixs import *
from  dataset import datasetCT,datasetCTall
import random
from utils.tools  import mkdir
from config import Config
from utils.WarmUpLR import WarmupLR
import torch.nn.functional as F

# set seed
GLOBAL_SEED = 1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def testAll(myModel,isTrain = True, isSave = False):
    '''
    test and get feature and uncertainty
    '''
    dataset = datasetCTall(Config.datapath,isOri=Config.isOri, isTraining=False, dataName=Config.dataName)
    data_loaderTest = DataLoader(dataset, batch_size=1)

    outf = Config.saveName
    if not os.path.isdir(Config.saveName):
        os.makedirs(Config.saveName)

    correct = 0
    total = 0
    f1 = open('%s/confidence_var.txt'%outf, 'w')
    f2 = open('%s/confidence_detail.txt'%outf, 'w')

    myModel.eval()
    myModel.apply(apply_dropout)
    output_np = []
    pre_all = []
    pre_01_all = []
    label_all = []
    with torch.no_grad():
        for j, validate_data in enumerate(data_loaderTest):
            print('No. %d/%d...' % (j,len(data_loaderTest)))
            inputs, labels = validate_data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, feature = myModel(inputs)
            feature = feature.cpu().detach().numpy()

            #save
            name = dataset.getFileName()
            path = os.path.join(Config.featSavePath, Config.dataName, name.split('/')[0], name.split('/')[1])
            if not os.path.isdir(path):
                os.makedirs(path)
            savePath = os.path.join(Config.featSavePath, Config.dataName, name[:-4])
            

            #uncertainty
            batch_output = []
            for k in range(Config.eva_iter):
                current_batch, _ = myModel(inputs)
                predicted_prob = torch.max(F.softmax(current_batch, dim=1))
                predicted_prob = predicted_prob.cpu().numpy()
                batch_output.append(predicted_prob)
            var = np.var(batch_output)
            mean = np.mean(batch_output)

            f1.write("{};{};{};\n".format(name,mean,var))
            f2.write("{};{};\n".format(name,np.array(batch_output)))

            #save feature and corresponding uncertainty
            np.savez(savePath+".npz", feature, var)

        f1.close()
        f2.close()
    return 0

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = Config.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = Config.savePath
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if Config.isOri:
        in_channel = 2
    else:
        in_channel = 1
    model = ResNet(in_ch=in_channel, num_classes=2).to(device)

    model = torch.nn.DataParallel(model).cuda()
   
    
    #get feature and uncertainty
    path = '/media/hjk/10E3196B10E3196B/dataSets/result/ChestCT/merge_resnet/state-89-306-AUC-0.7683444021769364.pth'

    model = torch.load(path)
    if isinstance(model,torch.nn.DataParallel):
        model = model.module

    testAll(model, isTrain=False)
