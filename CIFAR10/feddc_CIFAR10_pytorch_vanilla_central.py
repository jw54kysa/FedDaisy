import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import time
import os
import argparse

from dataset_cifar10 import *
from averaging import Average
from resnet import Cifar10ResNet50, Cifar10ResNet18
from client_pytorch import PyTorchNN, evaluateModel
from vanilla_training import trainEvalLoopVanilla
from createPlot import createLossAccPlot

parser = argparse.ArgumentParser(description='Federated daisy chaining')
# Training Hyperparameters
training_args = parser.add_argument_group('training')

training_args.add_argument('--optimizer', type=str, default='SGD', choices=['SGD','Adam'],
                    help='optimizer (default: SGD)')
training_args.add_argument('--train-batch-size', type=int, default=8,
                    help='input batch size for training (default: 8)')
parser.add_argument('--num-clients', type=int, default=1,
                    help='Number of clients in federated network (default: 1)')
parser.add_argument('--num-samples-per-client', type=int, default=8,
                    help='Number of samples each client has available (default: 8)')
parser.add_argument('--num-rounds', type=int, default=100,
                    help='Number of rounds of training (default: 100)')
training_args.add_argument('--lr-schedule-ep', type=int, default=None,
                    help='number of epochs after which to change lr (set to None if no schedule) (default: None)')
training_args.add_argument('--lr-change-rate', type=float, default=0.5,
                    help='(multiplicative) change factor for lr (default: 0.5)')
parser.add_argument('--report-rounds', type=int, default=25,
                    help='After how many rounds the model should be evaluated and performance reported (default: 25)')

parser.add_argument('--workers', type=int, default='16',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')

args = parser.parse_args()

torch.set_num_threads(args.workers)

randomState = args.seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

name = "Vanilla_Central"

mode = 'gpu'
lossFunction = "CrossEntropyLoss"

#here it would of course be smarter to have one GPU per client...
device = torch.device("cuda")

trainloader, testloader, classes = getCIFAR10DataLoader(args.train_batch_size, args.num_clients,
                                                        args.num_samples_per_client)

vanillaModel = Cifar10ResNet18().cuda(device)

if (lossFunction == "CrossEntropyLoss"):
    loss = nn.CrossEntropyLoss()
else:
    raise NotImplementedError

optimizer = eval("optim." + args.optimizer + "(vanillaModel.parameters(), lr=" + str(args.lr) + ")")

## Compute number of epochs
epochs = int(np.ceil(args.num_rounds / np.floor(len(trainloader.dataset) / args.num_samples_per_client)))

if (args.lr_schedule_ep is not None):
    ## adapt schedule
    schedule_ep = int(np.floor(args.lr_schedule_ep * (epochs / args.num_rounds)))

    scheduler = optim.lr_scheduler.StepLR(optimizer, schedule_ep, gamma=args.lr_change_rate, last_epoch=-1,
                                          verbose=False)
else:
    scheduler = None

trainEvalLoopVanilla(vanillaModel, loss, optimizer, scheduler, trainloader, testloader, epochs, device,
                     args.report_rounds)