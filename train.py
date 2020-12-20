import model as m
import dataset as d
from evaluator import Evaluator

import csv
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchvision import transforms
from pytorch_metric_learning import losses, miners, distances, reducers, testers
#from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

DATA_DIR = './CUB_200_2011'
TRAIN_TXT = './meta/CUB200/train.txt'
TEST_TXT = './meta/CUB200/test.txt'
BBOX_TXT = './meta/CUB200/bbox.txt'

CARS_DIR = './CARS_196'
CARS_MAT = './CARS_196/devkit/cars_annos.mat'


LCGD = False
loss_list = [[], []]
acc_list = [[], [], [], []]
pmn_list = [[], []]


def train(model, loss_func, aux_loss, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        aux, embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)

        loss1 = aux_loss(aux, labels)  # CE loss
        # triplet margin loss
        loss2 = loss_func(embeddings, labels, indices_tuple)

        loss = loss1 + loss2
        loss.backward()


        #custom derivative for p_k
        if LCGD:
            minn = 9999999
            maxx = -9999999
            for i in model.GDLayers:
                for p in i.parameters():
                    maxx = max(torch.max(p).item(), maxx)
                    minn = min(torch.min(p).item(), minn)

            print(minn, maxx)
            pmn_list[0].append(maxx)
            pmn_list[1].append(minn)        
        
                


        optimizer.step()
        if batch_idx % 20 == 0:
            print("Epoch {} Iteration {}: AuxLoss = {}, RankingLoss = {}, Number of mined triplets = {}".format(
                epoch, batch_idx, loss1, loss2, mining_func.num_triplets))
            loss_list[0].append(loss1)
            loss_list[1].append(loss2)


def argumentParsing():
    parser = argparse.ArgumentParser()

    defaultValue = {'lr': 1e-4,
                    'margin': 0.1,
                    'epoch': 3,
                    'batch': 128,
                    'M': 100,
                    'n': 1,
                    'T': 0.5,
                    'dim': 1536,
                    'd': '0',
                    'lcgd': '0', # 0 = cgd, else = initial p_k value of lcgd
                    'debug': '0', # 1 for anomaly detection
                    'eval': '1',
                    'gd': '1,3,inf'}
                    

    for arg in defaultValue:
        parser.add_argument('-'+arg, required=False)

    args = parser.parse_args()

    for i in vars(args):
        if vars(args)[i]:
            defaultValue[i] = vars(args)[i]

    return defaultValue

def write_csv(fileName, iterable):
    with open(fileName+'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(iterable)

if __name__ == "__main__":

    args = argumentParsing()

    print("Arguments:")
    print(args)

    device = torch.device("cuda")

    if args['debug'] == '1':
        torch.autograd.set_detect_anomaly(True)

    if args['d'] == '0':
        dataset = d.Dataset(DATA_DIR, TRAIN_TXT, TEST_TXT, bbox_txt=BBOX_TXT)
        dataset.print_stats()
    else:
        dataset = d.CARS_Dataset(CARS_DIR, CARS_MAT)
        dataset.print_stats()       

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = d.ImageData(dataset.train, transform)
    test_dataset = d.ImageData(dataset.test, transform)

    class_num = dataset.num_train_ids

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(args['batch']), shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(args['batch']))


    p_k_list = [float(x) for x in args['gd'].split(',')]
    
    #model = m.CGD(int(args['dim']), 1, int(args['M']), float(args['T']),  p_k_list).to(device)

    
    if args['lcgd'] == '0':
        model = m.CGD(int(args['dim']), 1, class_num,
                    float(args['T']),  p_k_list).to(device)
    else:
        LCGD = True
        model = m.LCGD(int(args['dim']), int(args['n']), class_num, float(args['T']), float(args['lcgd'])).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(args['lr']))

    distance = distances.LpDistance(power=2)
    reducer = reducers.ThresholdReducer(low=0)
    aux_loss = torch.nn.CrossEntropyLoss()
    loss_func = losses.TripletMarginLoss(margin=float(
        args['margin']), distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=float(
        args['margin']), distance=distance, type_of_triplets="hard")

    

    num_epochs = int(args['epoch'])
    for epoch in range(1, num_epochs+1):
        train(model, loss_func, aux_loss, mining_func,
              device, train_loader, optimizer, epoch)

        if args['eval'] == '1':
            #torch.cuda.empty_cache()
            print('test')
            model.eval()
            with torch.no_grad():
                evaluator = Evaluator(model, test_loader, device)
                recalls = evaluator.evaluate(ranks=[1, 2, 4, 8])
                for i, k in enumerate(recalls):
                    acc_list[i].append(k)

        
    torch.no_grad()

    print(acc_list)
    #plt.style.use('classic')

    #torch.cuda.empty_cache()
    if args['eval'] == '1':
        #write_csv('loss.csv', [x.item() for x in loss_list])
        #write_csv('recall.csv', acc_list)

        # loss plot
        plot1 = plt.figure(1)
        loss_name = ['aux loss', 'ranking loss']
        for i in range(len(loss_name)):
            plt.plot(loss_list[i], label=loss_name[i])
        plt.legend()
    
        # accuracy plot
        plot2 = plt.figure(2)
        acc_name = ['recall@1', 'recall@2', 'recall@4', 'recall@8']
        for i in range(len(acc_name)):
            plt.plot(acc_list[i], label=acc_name[i])
        plt.legend()

        # p_k plot
        if args['lcgd'] != '0':
            #write_csv('pk.csv', pmn_list)
            plot3 = plt.figure(3)
            pk_name = ['pk_max', 'pk_min']
            for i in range(len(pk_name)):
                plt.plot(pmn_list[i], label=pk_name[i])
            plt.legend()

        plt.show()

    else:
        model.eval()
        with torch.no_grad():
            evaluator = Evaluator(model, test_loader, device)
            recalls = evaluator.evaluate(ranks=[1, 2, 4, 8])
            print(recalls)
