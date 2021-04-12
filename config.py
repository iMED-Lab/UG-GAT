#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :config.py
@Description :
@Time        :2021/04/12 09:17:49
@Author      :Jinkui Hao
@Version     :1.0
'''

import os
class Config():

    #path
    datapath = '/media/hjk/10E3196B10E3196B/dataSets/PPE' 
    resultPath = '/media/hjk/10E3196B10E3196B/dataSets/result/ChestCT'
    featSavePath = '/media/hjk/10E3196B10E3196B/dataSets/PPE/5.featForGraph'

    saveName = 'Resnet_1' 
    savePath = os.path.join(resultPath, saveName)
    env = saveName

    batch_size = 4
    num_epochs = 500
    base_lr = 1e-4
    weight_decay = 0.0005

    eva_iter = 50

    isOri = True 
    dataName = 'merged-1'  

    #GPU
    gpu = '0,1'

class Config_graph():
    resultPath = '/media/hjk/10E3196B10E3196B/dataSets/result/ChestCT' 
    datapath = '/media/hjk/10E3196B10E3196B/dataSets/PPE'

    saveName = 'graph_1' 

    savePath = os.path.join(resultPath, saveName)
    env = saveName

    num_epochs = 100
    base_lr = 1e-5
    weight_decay = 0.0005

    dataName = 'merged-1'  

    #graph classification
    feat_in = 512  
    hidden = 256
    nclass = 2
    dropout = 0.2
    nb_heads = 4 #multi-head attention
    alpha = 0.2 #Alpha for the leaky_relu

    imgNum = 64

    #GPU
    gpu = '1'
