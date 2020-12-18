import model as m
import dataset as d
from evaluator import Evaluator

import argparse
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from pytorch_metric_learning import losses, miners, distances, reducers, testers
#from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

DATA_DIR = './CUB_200_2011'
TRAIN_TXT = './meta/CUB200/train.txt'
TEST_TXT = './meta/CUB200/test.txt'


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, loss, mining_func.num_triplets))

'''
def get_all_embeddings(dataset, model, device):
    tester = testers.BaseTester(data_device=device)
    return tester.get_all_embeddings(dataset, model)

def test(train_set, test_set, model, device, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model, device)
    test_embeddings, test_labels = get_all_embeddings(test_set, model, device)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
                                                train_embeddings,
                                                np.squeeze(test_labels),
                                                np.squeeze(train_labels),
                                                False)
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
'''

def argumentParsing():
    parser = argparse.ArgumentParser()
    
    defaultValue = {'lr': 1e-4,
                    'margin': 0.1,
                    'epoch': 3,
                    'batch': 128,
                    'dim': 1536,
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

    device=torch.device("cuda")

    dataset = d.Dataset(DATA_DIR, TRAIN_TXT, TEST_TXT)
    dataset.print_stats()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_dataset = d.ImageData(dataset.train, transform)
    test_dataset = d.ImageData(dataset.test, transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args['batch']), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(args['batch']))

    p_k_list = [float(x) for x in args['gd'].split(',')]

    model = m.CGD(int(args['dim']), 1, args['batch'], p_k_list).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(args['lr']))

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low = 0)
    loss_func = losses.TripletMarginLoss(margin = float(args['margin']), distance = distance, reducer = reducer)
    mining_func = miners.TripletMarginMiner(margin = float(args['margin']), distance = distance, type_of_triplets = "hard")
    # accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)

    num_epochs = int(args['epoch'])
    for epoch in range(1, num_epochs+1):
        train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)

    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        evaluator = Evaluator(model, test_loader, device)
        recalls = evaluator.evaluate(ranks=[1, 2, 4, 8])
        print(recalls)