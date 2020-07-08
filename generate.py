import model
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def getLatentCoordinates(network, data, batchSize=None):
    network = network.cuda()

    data = torch.from_numpy(data).cuda()
    dataset = TensorDataset(data)
    batchSize = len(dataset) if batchSize is None else batchSize
    dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=False)

    out = np.empty((0,10))
    for data in dataLoader:
        data = data[0]
        output = network(data).cpu().data.numpy()
        out = np.concatenate((out, output))

    return out

def generateImages(n, network, latentSize = 10):
    network = network.cuda()

    noise = torch.rand((n,latentSize)).cuda()*10 - 5

    imgs = network(noise)
    imgs = imgs.view(imgs.shape[0],28,28)    
    imgs = imgs.cpu().data.numpy()

    return imgs