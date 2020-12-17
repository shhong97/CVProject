import model as m
import dataset as d

import argparse
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

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


def argumentParsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', required=False)
    parser.add_argument('-margin', required=False)
    parser.add_argument('-epoch', required=True)
    parser.add_argument('-batch', required=False)

    args = parser.parse_args()

    if args.lr is not None:
        lr = float(args.lr)
    else:
        lr = 1e-4

    if args.margin is not None:
        margin = float(args.margin)
    else:
        margin = 0.1

    if args.batch is not None:
        batch_size = int(args.batch)
    else:
        batch_size = 256


    epoch = int(args.epoch)

    return lr, margin, epoch, batch_size


if __name__ == "__main__":

    lr, margin, num_epochs, batch_size = argumentParsing()

    device=torch.device("cuda")

    dataset = d.Dataset(DATA_DIR, TRAIN_TXT, TEST_TXT)
    dataset.print_stats()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])


    train_dataset = d.ImageData(dataset.train, transform)
    test_dataset = d.ImageData(dataset.test, transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


    model = m.CGD(1536, 1, 1024, [1, 2, 3]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low = 0)
    loss_func = losses.TripletMarginLoss(margin = margin, distance = distance, reducer = reducer)
    mining_func = miners.TripletMarginMiner(margin = margin, distance = distance, type_of_triplets = "hard")
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)

    for epoch in range(1, num_epochs+1):
        train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        #test(train_dataset, test_dataset, model, device, accuracy_calculator)