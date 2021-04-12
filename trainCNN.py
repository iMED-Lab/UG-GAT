#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :trainCNN.py
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

def get_lr(optier):
    for param_group in optier.param_groups:
        return param_group['lr']

def test(myModel,isTrain = True, isSave = True):
    if isSave:
        dataset = datasetCT(Config.datapath,isOri=Config.isOri, isTraining=False, dataName=Config.dataName)
        data_loaderTest = DataLoader(dataset, batch_size=3)
    else:
        dataset = datasetCT(Config.datapath,isOri=Config.isOri, isTraining=False, dataName=Config.dataName)
        data_loaderTest = DataLoader(dataset, batch_size=32)


    myModel.eval()
    correct = 0
    correct_another = 0
    total = 0
    output_np = []
    pre_all = []
    pre_01_all = []
    label_all = []
    csvFile = open(Config.savePath + "/test_result.csv", "w")
    with torch.no_grad():
        for j, validate_data in enumerate(data_loaderTest):
            print('No. %d/%d...' % (j,len(data_loaderTest)))
            inputs, labels = validate_data
            inputs, labels = inputs.to(device), labels.to(device)
            # labels = torch.Tensor(labels)
            outputs,_ = myModel(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            output_np = [output_np, outputs.cpu().detach().numpy()]

            #value, predicted = torch.max(outputs.data, 1)

            value = outputs[:,1]
            threshhold = 0.5
            #大于 threshhold
            zero = torch.zeros_like(value)
            one = torch.ones_like(value)
            predicted = torch.where(value > threshhold, one, zero)
            inputs, labels = inputs.to(device), labels.to(device)

            value1 = value.cpu().detach().numpy()
            labels1 = labels.cpu().detach().numpy()
            predicted1 = predicted.cpu().detach().numpy()

            # if isTrain is False:
            #     if predicted1 == 0:
            #         value1 = -value1
            pre_all = np.append(pre_all, value1)
            label_all = np.append(label_all, labels1)
            pre_01_all = np.append(pre_01_all, predicted1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # outputs = numpy.argmax(outputs.cpu().data.numpy(), axis=1)
            # equal = outputs.reshape([-1, 1]) == labels.cpu().data.numpy().reshape([-1, 1])

           
            name = dataset.getFileName()
            writer = csv.writer(csvFile)

            value, predicted = value.cpu().detach().numpy(), predicted.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            data = [name, predicted[0], labels[0]]
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

    myModel.train(mode=True)
    csvFile.close()

    return 100.0 * correct / total , output_np,AUC,Acc,Sen,Spe

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

def train(myModel,optimizer,criterion,num_epochs,batch_size):

    data_loaderTrain = DataLoader(datasetCT(Config.datapath,isOri=Config.isOri, isTraining=True, dataName=Config.dataName), batch_size=batch_size,shuffle=True, worker_init_fn=worker_init_fn)

    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs)
    schedulers = WarmupLR(scheduler_steplr, init_lr=0, num_warmup=5, warmup_strategy='cos')

    maxValue = {'acc': 0}
    best_auc = 0.0
    save = True
    #print(len(data_loaderTrain.dataset))

    for epoch in range(num_epochs):
        schedulers.step(epoch)
        print('Epoch %d/%d' % (epoch, num_epochs - 1))
        print('-' * 10)
        runing_loss = 0.0
        myModel.train(mode=True)
        for i, data in enumerate(data_loaderTrain):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape)

            vis.img(name='images', img_=inputs[0, 0, :, :])
            optimizer.zero_grad()

            outputs,_ = myModel(inputs)

            #print("shape of output:",outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            # optimizer1.step()
            optimizer.step()
            # print statistics
            runing_loss += loss.item()
            print('-' * 10)
            vis.plot('train_main_loss', runing_loss/(i+1))
            
            #test_acc, prediction, AUC, Acc, Sen, Spe = test(myModel,isTrain=False, isSave=False)
            print("%d/%d,train_loss:%0.4f" % (i, (len(data_loaderTrain.dataset) - 1) // data_loaderTrain.batch_size + 1, runing_loss/(i+1)))

        print('Epoch %d/%d' % (epoch, num_epochs - 1))
        current_lr = get_lr(optimizer)
        vis.plot('learning rate', current_lr)
        if bool(epoch % 4) is False:
            #trainAcc, _, _, _, _, _ = test(myModel, isTrain=True)
            if bool(epoch % 5) is False:
                save = True
            else:
                save = False
            test_acc, prediction, AUC, Acc, Sen, Spe = test(myModel,isTrain=False, isSave=save)
            vis.plot('Test AUC', AUC)
            vis.plot('Test Acc', Acc)
            vis.plot('Test SEN', Sen)
            vis.plot('Test SPE', Spe)

            
            isExists = os.path.exists(save_dir)
            if not isExists:
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'state-{}-{}-AUC-{}.pth'.format(epoch + 1, i + 1,AUC))

            if AUC > best_auc-0.03:
                best_auc = AUC
                torch.save(myModel, save_path)
            if AUC >= 0.75:
                torch.save(myModel, save_path)

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
    myModel = ResNet(in_ch=in_channel, num_classes=2).to(device)

    myModel = torch.nn.DataParallel(myModel).cuda()

    optimizer = optim.Adam(myModel.parameters(), lr=Config.base_lr, weight_decay=Config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best = {'loss': 0.0, 'save': ''}
    vis = Visualizer(env=Config.env)
    
    train(myModel, optimizer, criterion, Config.num_epochs,Config.batch_size)
