import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from create_data import loadTest, loadTrain, Data, collate_fn, dileImage
from functionCi import *
from model import ImageNet
# training function at each epoch

def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    loss = 0
    for batch_idx, data in enumerate(train_loader):
        x = data[0].to(device)
        y = data[1].to(device)
        target = data[2].to(device)
        x = x.to(device)

        optimizer.zero_grad()
        model.to(device)
        output = model(x,target)
        loss = loss_fn(output, y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return loss.item()


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            x = data[0].to(device)
            y = data[1].to(device)
            target = data[2].to(device)


            model.to(device)
            output = model(x, target)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()





torch.backends.cudnn.enabled = False

datasets = [['davis','kiba'][int(sys.argv[1])]]

cuda_name = "cuda:0"
if len(sys.argv)>2:

    cuda_name = "cuda:" + str(int(sys.argv[2]))
print('cuda_name:', cuda_name)
print(torch.__version__)
print(torch.cuda.is_available())


TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    train_data = loadTrain(dataset)
    test_data = loadTest(dataset)

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    checkpoint_path = 'checkpoint.pt'
    startEpoch = 0

    # training the model
    device = torch.device(cuda_name)
    model = ImageNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startEpoch = checkpoint['epoch']

    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = 'model_' + '_' + dataset +  '.pth'
    result_file_name = 'result_' + '_' + dataset +  '.csv'
    for epoch in range(startEpoch,NUM_EPOCHS):
        t_loss = train(model, device, train_loader, optimizer, epoch+1)
        G,P= predicting(model, device, test_loader)
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },checkpoint_path)
        if ret[1]<best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name,'w') as f:
                f.write(','.join(map(str,ret)))
            best_epoch = epoch+1
            best_mse = ret[1]
            best_ci = ret[4]

            print('ci improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,dataset)
        else:
            print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,dataset)

