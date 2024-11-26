import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils
import numpy as np
import random

from collections import Counter

def getCIFAR10(device):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)


    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

                                   

    X_train, y_train = trainset.data, np.array(trainset.targets)  #next(iter(trainloader))[0].numpy(), next(iter(trainloader))[1].numpy()
    X_test, y_test   = testset.data, np.array(testset.targets)    #next(iter(testloader))[0].numpy(), next(iter(testloader))[1].numpy()
    #X_train = X_train.reshape(X_train.shape[0],3,32,32)
    #X_test = X_test.reshape(X_test.shape[0],3,32,32)
    X_train = torch.cuda.FloatTensor(X_train).permute(0,3,1,2)
    y_train = torch.tensor(y_train, device=device)
    X_test = torch.cuda.FloatTensor(X_test).permute(0,3,1,2)
    y_test = torch.tensor(y_test, device=device)
    return X_train, y_train, X_test, y_test


def splitIntoLocalData(n, m, n_local, rng):
    if m*n_local > n:
        print("Error: not enough data (n=",n,") for",m,"sets of size ",n_local,". Reducing local size to",n//m,".")
        n_local = n // m
        
    idxs = np.arange(n)
    rng.shuffle(idxs)
    bucket_size = n_local
    i = 0
    client_idxs = []
    while (i+1)*bucket_size <= n and i < m:
        idx = idxs[i*bucket_size:(i+1)*bucket_size]
        client_idxs.append(idx)
        i += 1
    return client_idxs

# split data into different-sized samples
# n len(testdata)
# m number of clients
# n_local_min min number of samples per client
# n_local_max max number of samples per client
def splitIntoLocalDataRandLen(n, m, n_local_min, n_local_max, rng):
    if m * n_local_max > n:
        print("Error: not enough data (n=", n, ") for", m, "sets of size ",
              n_local_max, ". Reducing local size to", n // m, ".")
        n_local_max = n // m

    idxs = np.arange(n)
    rng.shuffle(idxs)
    i = 0
    client_idxs = []
    while (i + 1) * n_local_max <= n and i < m:
        bucket_size = rng.randint(n_local_min, n_local_max)
        idx = idxs[i * n_local_max:(i * n_local_max) + bucket_size]
        client_idxs.append(idx)
        i += 1
    return client_idxs

# Probabilistic permutation for localDataIndex
# samples localDataIndex with respect to samples size
# larger sample sets are selected with higher probability
def localDataIndexRandLenPermutation(localDataIndex, client_idxs, with_amp):
    n = len(localDataIndex)
    if len(client_idxs) != n:
        print("localDataIndex, client_idxs not same length !!!")
        return None

    # Calculate sample sizes
    sample_sizes = np.array([len(client_idxs[i]) for i in range(n)])

    if with_amp:  # Amplify higher-ranked indices by squaring their sample sizes
        sample_sizes = sample_sizes ** 2

    # Generate probabilities proportional to sample sizes
    total_size = sample_sizes.sum()
    if total_size == 0:
        print("Total sample size is zero, cannot generate probabilities!")
        return None

    probabilities = (sample_sizes / total_size)

    permutedDataIndex = np.random.choice(range(n), size=n, replace=True, p=probabilities)
    return permutedDataIndex

##TODO: getSample that cycles through all local data batches and only shuffles batches after one epoch. 

def getSample(idxs_of_client, batchSize, rng):
    n = len(idxs_of_client)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    sample_idxs = np.array([idxs_of_client[idxs[j]] for j in range(batchSize)])
    return sample_idxs


def splitIntoLocalDataLimClasses(X, y, m, n_local, rng, numClasses):

    ## generate different stacks of samples, one per class
    idcs = [ np.array([yIdx for yIdx, yVal in enumerate(y) if yVal == i]) for i in range(10) ]
    for i in range(10):
        rng.shuffle(idcs[i])

    ## offsets in these stacks
    idcsOffsets = np.repeat(0,10)

    client_idxs = []
    for i in range(m):

        ## determine which classes to cover
        class_pick = np.arange(10)
        rng.shuffle(class_pick)
        class_pick = class_pick[:numClasses]
        ## generate assignment of samples to classes (uniform at random)
        classIdcs = [ class_pick[c] for c in np.random.random_integers(0, numClasses-1, n_local) ]
        ## get the sample indices
        sample = np.concatenate([idcs[cl][idcsOffsets[cl]:(idcsOffsets[cl]+count)] for cl, count in Counter(classIdcs).items()])

        client_idxs.append(sample)

        for cl, count in Counter(classIdcs).items():
            idcsOffsets[cl] += count

    return client_idxs


def getCIFAR10DataLoader(batch_size, num_clients, n_local):

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    n = len(trainset)
    trainset = torch.utils.data.Subset(trainset,
        random.sample(range(n), min(num_clients*n_local, n))
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes
