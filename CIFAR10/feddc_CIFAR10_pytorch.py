
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

#set the parameters

parser = argparse.ArgumentParser(description='Federated daisy chaining')
# Training Hyperparameters
training_args = parser.add_argument_group('training')
training_args.add_argument('--dataset', type=str, default='mnist',
                    choices=['cifar10'],
                    help='dataset (default: Cifar10)')
training_args.add_argument('--model', type=str, default='resnet18', choices=['resnet18'],
                    help='model architecture (default: resnet18)')
training_args.add_argument('--optimizer', type=str, default='SGD', choices=['SGD','Adam'],
                    help='optimizer (default: SGD)')
training_args.add_argument('--train-batch-size', type=int, default=8,
                    help='input batch size for training (default: 8)')
training_args.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
training_args.add_argument('--lr-schedule-ep', type=int, default=None,
                    help='number of epochs after which to change lr (set to None if no schedule) (default: None)')
training_args.add_argument('--lr-change-rate', type=float, default=0.5,
                    help='(multiplicative) change factor for lr (default: 0.5)')

parser.add_argument('--num-clients', type=int, default=1,
                    help='Number of clients in federated network (default: 1)')
parser.add_argument('--num-rounds', type=int, default=100,
                    help='Number of rounds of training (default: 100)')
parser.add_argument('--num-samples-per-client', type=int, default=8,
                    help='Number of samples each client has available (default: 8)')
parser.add_argument('--report-rounds', type=int, default=25,
                    help='After how many rounds the model should be evaluated and performance reported (default: 25)')
parser.add_argument('--daisy-rounds', type=int, default=1,
                    help='After how many rounds daisy chaining should be used to communicate models (default: 1)')
parser.add_argument('--aggregate-rounds', type=int, default=10,
                    help='After how many rounds the models should be aggregated by the coordinator (default: 10)')

parser.add_argument('--restrict-classes', type=int, default=None,
                    help='Number of classes that should be available at maximum for individual clients, if None then all classes are available (default: None)')

parser.add_argument('--run-ablation', type=str, default=None,
                    choices=['vanilla_training'],
                    help='Type of ablation to run (default: None)')

parser.add_argument('--iid-data', type=str, default=None,
                    choices=['randsize'],
                    help='type of local dataset manipulation')
parser.add_argument('--min-samples', type=int, default=8,
                    help='Minimal number of samples each client has available (default: 8)')
parser.add_argument('--max-samples', type=int, default=64,
                    help='Maximal number of samples each client has available (default: 64)')
parser.add_argument('--rand', type=str, default='rand',
                    choices=['rand', 'prob'],
                    help='Type of daisy round permutation.')
parser.add_argument('--with-amp', type=str, default='yes',
                    help='Amplification of probabilistic daisy round permutation.')


## Experiment Hyperparameters ##
# parser.add_argument('--expid', type=str, default='',
#                     help='name used to save results (default: "")')
# parser.add_argument('--result-dir', type=str, default='Results/data',
#                     help='path to directory to save results (default: "Results/data")')
parser.add_argument('--workers', type=int, default='16',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
# parser.add_argument('--verbose', action='store_true',
#                     help='print statistics during training and testing')


args = parser.parse_args()


torch.set_num_threads(args.workers)

randomState = args.seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

iid_data_info=""
if args.iid_data == 'randsize':
    iid_data_info = f"iid{args.iid_data}{args.min_samples}_{args.max_samples}_per{args.permutation}_"
name = f"FedDC_cifar10_{iid_data_info}nc{args.num_clients}cl_n{args.num_samples_per_client}_b{args.train_batch_size}_d{args.daisy_rounds}_a{args.aggregate_rounds}_lr0_schedule{args.lr_schedule_ep}_r{args.num_rounds}_s{args.seed}"

aggregator = Average()
mode = 'gpu'
lossFunction = "CrossEntropyLoss"


#here it would of course be smarter to have one GPU per client...
device = torch.device("cuda") #torch.device("cuda:0" if torch.cuda.is_available() else None)

if (args.run_ablation is None):

    #initialize clients
    clients = []
    for _ in range(args.num_clients):
        client = PyTorchNN(args.train_batch_size, mode, device)
        torchnetwork = Cifar10ResNet18() #Cifar10ResNet50()
        torchnetwork = torchnetwork.cuda(device)
        client.setCore(torchnetwork)
        client.setLoss(lossFunction)
        client.setUpdateRule(args.optimizer, args.lr, args.lr_schedule_ep, args.lr_change_rate)
        clients.append(client)
        
    #get a fixed random number generator
    rng = np.random.RandomState(randomState)

    #set up a folder for logging
    exp_path = name + "_" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    os.mkdir(exp_path)

    #log basic experiment properties
    f = open(exp_path+"/setup.txt",'w')
    out  = "aggregator = "+str(aggregator)+"\n"
    out += "m = "+str(args.num_clients)+"\nn_local = "+str(args.num_samples_per_client)+"\n"
    out += "d = "+str(args.daisy_rounds)+"\nb = "+str(args.aggregate_rounds)+"\n"
    out += "rounds = "+str(args.num_rounds)+"\n"
    out += "batchSize = "+str(args.train_batch_size)+"\n"
    out += "updateRule = "+str(args.optimizer)+"\nlearningRate = "+str(args.lr)+"\n"
    out += "lossFunction = "+str(lossFunction)+"\n"
    out += "randomState = "+str(randomState)+"\n"

    out += f"""
iid-data = {args.iid_data}
min-samples = {args.min_samples}
max-samples = {args.max_samples}
permutation = {args.permutation}
with-amp = {args.with_amp}    
    """
    f.write(out)
    f.close()

    #get the data
    X_train, y_train, X_test, y_test = getCIFAR10(device)
    n_train = y_train.shape[0]
    if (args.restrict_classes is None):
        if args.iid_data == 'randsize':
            client_idxs = splitIntoLocalDataRandLen(n_train, args.num_clients, args.min_samples, args.max_samples, rng)
        else:
            client_idxs = splitIntoLocalData(n_train, args.num_clients, args.num_samples_per_client, rng)
    else:
        client_idxs = splitIntoLocalDataLimClasses(X_train, y_train, args.num_clients, args.num_samples_per_client, rng, args.restrict_classes)

    localDataIndex = np.arange(args.num_clients)
    withAmp = True if args.with_amp == 'yes' else False

    trainLosses = [[] for _ in range(args.num_clients)]
    testLosses = [[] for _ in range(args.num_clients)]
    trainACCs = [[] for _ in range(args.num_clients)]
    testACCs = [[] for _ in range(args.num_clients)]

    ## TODO: Move everything (including data) to GPU and only work with indices here.
    for t in range(args.num_rounds):
        for i in range(args.num_clients):
            if args.iid_data == 'randsize':
                # get random length data samples
                sample = getSample(client_idxs[localDataIndex[i]], len(client_idxs[localDataIndex[i]]), rng) # args.train_batch_size
            else:
                sample = getSample(client_idxs[localDataIndex[i]], args.train_batch_size, rng)
            clients[i].update(sample, X_train, y_train) ##TODO: sample is now a list of indices
        if t % args.daisy_rounds == args.daisy_rounds - 1: #daisy chaining

            if args.permutation == 'prob':
                localDataIndex = localDataIndexRandLenPermutation(localDataIndex, client_idxs, withAmp)
            else:
                rng.shuffle(localDataIndex)

        if t % args.aggregate_rounds == args.aggregate_rounds - 1: #aggregation
            params = []
            for i in range(args.num_clients): #get the model parameters (weights) of all clients
                params.append(clients[i].getParameters())
            aggParams = aggregator(params) #compute the aggregate
            for i in range(args.num_clients): 
                clients[i].setParameters(aggParams) #give each client the aggregate as new parameters

        #compute the train and test loss for each client (we might have to do that a bit more rarely for efficiency reasons)
        if t % args.report_rounds == 0:
            for i in range(args.num_clients):
                trainloss, trainACC, testloss, testACC = evaluateModel(clients[i], client_idxs[localDataIndex[i]], X_train, y_train, X_test, y_test)
                if mode == 'gpu':
                    trainloss = trainloss.cpu()
                    testloss = testloss.cpu()
                trainLosses[i].append(trainloss.numpy())
                testLosses[i].append(testloss.numpy())
                trainACCs[i].append(trainACC)
                testACCs[i].append(testACC)
            print("average train loss = ",np.mean(trainLosses[-1]), " average test loss = ",np.mean(testLosses[-1]))
            print("average train accuracy = ",np.mean(trainACCs[-1]), " average test accuracy = ",np.mean(testACCs[-1]))

    pickle.dump(trainLosses, open(exp_path+"/trainLosses.pck",'wb'))
    pickle.dump(testLosses, open(exp_path+"/testLosses.pck",'wb'))
    pickle.dump(trainACCs, open(exp_path + "/trainACCs.pck", 'wb'))
    pickle.dump(testACCs, open(exp_path + "/testACCs.pck", 'wb'))

    createLossAccPlot(exp_path)

elif (args.run_ablation == 'vanilla_training'):

    trainloader, testloader, classes = getCIFAR10DataLoader(args.train_batch_size, args.num_clients, args.num_samples_per_client)

    vanillaModel = Cifar10ResNet18().cuda(device)

    if (lossFunction == "CrossEntropyLoss"):   
        loss = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
        
    optimizer = eval("optim." + args.optimizer + "(vanillaModel.parameters(), lr=" + str(args.lr)  + ")")

    ## Compute number of epochs
    epochs = int(np.ceil(args.num_rounds/np.floor(len(trainloader.dataset)/args.num_samples_per_client)))


    if (args.lr_schedule_ep is not None):
        ## adapt schedule
        schedule_ep = int(np.floor(args.lr_schedule_ep * (epochs/args.num_rounds)))

        scheduler = optim.lr_scheduler.StepLR(optimizer, schedule_ep, gamma=args.lr_change_rate, last_epoch=-1, verbose=False)
    else:
        scheduler = None


    trainEvalLoopVanilla(vanillaModel, loss, optimizer, scheduler, trainloader, testloader, epochs, device, args.report_rounds)


