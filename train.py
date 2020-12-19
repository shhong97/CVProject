import model as m
import dataset as d
from evaluator import Evaluator

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

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
            for i in model.GDLayers:
                for p in i.parameters():
                    print (torch.min(p).item(), torch.max(p).item())
        
                


        optimizer.step()
        if batch_idx % 20 == 0:
            print("Epoch {} Iteration {}: AuxLoss = {}, RankingLoss = {}, Number of mined triplets = {}".format(
                epoch, batch_idx, loss1, loss2, mining_func.num_triplets))


def argumentParsing():
    parser = argparse.ArgumentParser()

    defaultValue = {'lr': 1e-4,
                    'margin': 0.1,
                    'epoch': 3,
                    'batch': 128,
                    'M': 100,
                    'T': 0.5,
                    'dim': 1536,
                    'd': '0',
                    'lcgd': '0', # 0 = cgd, else = initial p_k value of lcgd
                    'gd': '1,3,inf'}
                    

    for arg in defaultValue:
        parser.add_argument('-'+arg, required=False)

    args = parser.parse_args()

    for i in vars(args):
        if vars(args)[i]:
            defaultValue[i] = vars(args)[i]

    return defaultValue


if __name__ == "__main__":

    args = argumentParsing()

    print("Arguments:")
    print(args)

    device = torch.device("cuda")

    #torch.autograd.set_detect_anomaly(True)

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

    for i in train_loader:
        print (i)


    p_k_list = [float(x) for x in args['gd'].split(',')]
    
<<<<<<< Updated upstream

<<<<<<< Updated upstream

    #model = m.CGD(int(args['dim']), 1, int(args['M']), float(args['T']),  p_k_list).to(device)

    


    if args['lcgd'] == '0':
        model = m.CGD(int(args['dim']), 1, class_num,
                    float(args['T']),  p_k_list).to(device)
    else:
        LCGD = True
        model = m.LCGD(int(args['dim']), 1, class_num, float(args['T']), float(args['lcgd'])).to(device)

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

    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        evaluator = Evaluator(model, test_loader, device)
        recalls = evaluator.evaluate(ranks=[1, 2, 4, 8])
        print(recalls)
