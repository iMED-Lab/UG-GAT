#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :trainGraph.py
@Description : training and testing of UGGAT
@Time        :2021/04/12 09:39:04
@Author      :Jinkui Hao
@Version     :1.0
'''


from config import Config_graph
from UGGAT.models import *
from dataset import datasetGraphCla
import torch
from utils.WarmUpLR import WarmupLR
from utils.Visualizer import Visualizer
import os
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
from torch import optim, nn
import csv
from evaluation.matrixs import *

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

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv) 
    mx = r_mat_inv.dot(mx)
    return mx 

def get_lr(optier):
    for param_group in optier.param_groups:
        return param_group['lr']

def test(network, data_loaderTest, adj,epoch):
    
    network.eval()
    correct = 0
    correct_another = 0
    total = 0
    output_np = []
    pre_all = []
    pre_01_all = []
    label_all = []
    csvFile = open(Config_graph.savePath + "/%d_TestResult.csv"%epoch, "w")

    with torch.no_grad():
        for j, data in enumerate(data_loaderTest, 0):
            print('No. %d/%d...' % (j,len(data_loaderTest)))

            inputs, uncertainty, labels = data
            inputs = inputs.squeeze()
            uncertainty = uncertainty.squeeze()
            inputs, uncertainty, labels = inputs.to(device), uncertainty.to(device), labels.to(device)

            outputs = network(inputs, adj, uncertainty)
            outputs = nn.Softmax(dim=1)(outputs)

            value = outputs[:,1]
            threshhold = 0.5
            zero = torch.zeros_like(value)
            one = torch.ones_like(value)
            predicted = torch.where(value > threshhold, one, zero)
            inputs, labels = inputs.to(device), labels.to(device)

            value1 = value.cpu().detach().numpy()
            labels1 = labels.cpu().detach().numpy()
            predicted1 = predicted.cpu().detach().numpy()

            pre_all = np.append(pre_all, value1)
            label_all = np.append(label_all, labels1)
            pre_01_all = np.append(pre_01_all, predicted1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            name = data_loaderTest.dataset.getFileName()
            writer = csv.writer(csvFile)

            value, predicted = value.cpu().detach().numpy(), predicted.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            data = [name, value1[0], predicted[0], labels[0]]
            # 写入数据
            writer.writerow(data)

    print('Accuracy of the network on the  test images: %.3f %%' % (100.0 * correct / total))
    AUC = AUC_score(pre_all,label_all)
    threshhold = 0.5
    #print('threshhold:',threshhold)
    pre_01_all[pre_all >= threshhold] = 1
    pre_01_all[pre_all < threshhold] = 0

    Acc = accuracy_score(pre_01_all, label_all)
    Sen = recall_score(pre_01_all, label_all)
    Spe = specificity_score(pre_01_all, label_all)
    csvFile.close()

    return 100.0 * correct / total , output_np,AUC,Acc,Sen,Spe

def train(network, dataloader,dataloader_test, adj):
    # Create Optimizer
    lrate = Config_graph.base_lr
    optimizer = optim.Adam(network.parameters(), lr = lrate)
    criterion = torch.nn.CrossEntropyLoss()

    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=Config_graph.num_epochs)
    schedulers = WarmupLR(scheduler_steplr, init_lr=1e-7, num_warmup=5, warmup_strategy='cos')

    # Train model on the dataset
    for epoch in range(Config_graph.num_epochs):
        schedulers.step(epoch)
        print('Epoch %d/%d' % (epoch, Config_graph.num_epochs - 1))
        print('-' * 10)
        runing_loss = 0.0
        network.train(mode=True)

        for i, data in enumerate(dataloader, 0):

            inputs, uncertainty, labels = data
            inputs = inputs.squeeze()
            uncertainty = uncertainty.squeeze()
            inputs, uncertainty, labels = inputs.to(device), uncertainty.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = network(inputs, adj, uncertainty)

            loss = criterion(outputs, labels)
           
            #loss = my_pts_loss
            loss.backward()
            optimizer.step()
            runing_loss += loss.item()
            print('-' * 10)
            vis.plot('train_main_loss', runing_loss/(i+1))
            print("%d/%d,train_loss:%0.4f" % (i, (len(data_loaderTrain.dataset) - 1) // data_loaderTrain.batch_size + 1, runing_loss/(i+1)))


        print('Epoch %d/%d' % (epoch, Config_graph.num_epochs - 1))
        current_lr = get_lr(optimizer)
        vis.plot('learning rate', current_lr)
        if bool(epoch % 1) is False:
            if bool(epoch % 5) is False:
                save = True
            else:
                save = False
            test_acc, prediction, AUC, Acc, Sen, Spe = test(network,dataloader_test,adj,epoch)
            vis.plot('Test AUC', AUC)
            vis.plot('Test Acc', Acc)
            vis.plot('Test SEN', Sen)
            vis.plot('Test SPE', Spe)

            
            isExists = os.path.exists(save_dir)
            if not isExists:
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'state-{}-{}-AUC-{}.pth'.format(epoch + 1, i + 1,AUC))
            if AUC >= 0.75:
                torch.save(network, save_path)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = Config_graph.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Config_graph.savePath
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    vis = Visualizer(env=Config_graph.env)
    adjMetrix = np.load('utils/adjMetrix.npy')
    adjMetrix = torch.from_numpy(adjMetrix)
    adjMetrix = sp.coo_matrix(adjMetrix)
    adjMetrix = normalize(adjMetrix + sp.eye(adjMetrix.shape[0])) 
    adjMetrix = adjMetrix.todense()
    adjMetrix = torch.from_numpy(adjMetrix)
    adjMetrix = adjMetrix.float()

    adjMetrix = adjMetrix.to(device)

    model = UGGAT(nfeat=Config_graph.feat_in, 
                    nhid=Config_graph.hidden, 
                    nclass=Config_graph.nclass, 
                    dropout=Config_graph.dropout, 
                    nheads=Config_graph.nb_heads, 
                    alpha=Config_graph.alpha)

    model = model.to(device)
    datasetTrain = datasetGraphCla(Config_graph.datapath,isTraining=True, imgNum = Config_graph.imgNum,dataName=Config_graph.dataName)
    data_loaderTrain = DataLoader(datasetTrain, batch_size=1,shuffle=True, worker_init_fn=worker_init_fn)

    datasetTest = datasetGraphCla(Config_graph.datapath,isTraining=False, imgNum = Config_graph.imgNum,dataName=Config_graph.dataName)
    data_loaderTest = DataLoader(datasetTest, batch_size=1,shuffle=True, worker_init_fn=worker_init_fn)
    
    train(model, data_loaderTrain, data_loaderTest, adjMetrix)
