import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import model
import torch
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
sns.set()

def readData(dataDir="data/"):
    trainX = np.load(dataDir + "trainImg.npy").astype('float32')
    trainY = np.load(dataDir + "trainLbl.npy").astype('int64')
    testX = np.load(dataDir + "testImg.npy").astype('float32')
    testY = np.load(dataDir + "testLbl.npy").astype('int64')
    trainX /= 255.0
    testX /= 255.0

    #reshape for convolutional architecture
    trainXCon = trainX.reshape(trainX.shape[0],1,trainX.shape[1],trainX.shape[2])
    testXCon = testX.reshape(testX.shape[0],1,testX.shape[1],testX.shape[2])

    return trainXCon, trainY, testXCon, testY

top = cm.get_cmap('Reds_r', 128)    #define new colormap for drawing 
bottom = cm.get_cmap('Greys', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128)))) #first half is scale of reds, second is scale of grays
mnistColours = ListedColormap(newcolors, name='SplitMNIST')

def plotImage(img, ax, allowNegatives, normaliseImage):
    if len(img.shape) == 3:
        img = img[0]
    if normaliseImage:
        vmax = 1
        vmin = -1 if allowNegatives else 0 
    else:
        vmax = max(np.amax(img),-np.amin(img))
        vmin = -vmax if allowNegatives else 0
    
    cmap = mnistColours if allowNegatives else cm.get_cmap('Greys', 128)
    ax.imshow(img, cmap = cmap, aspect="equal", vmin=vmin,vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.grid(False)

def plotImages(array, nrows, ncols, allowNegatives, normaliseImage = True):
    f,axes = plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True)

    for row in range(nrows):
        for col in range(ncols):
            plotImage(array[row*ncols + col], axes[row][col],allowNegatives, normaliseImage)
    
    plt.show()
    return f,axes

def loadModel(path):
    encoderDecoder = model.EncoderDecoder()
    encoderDecoder.load_state_dict(torch.load(path))
    return encoderDecoder


def plotPCA(x, y, frequency = 10):
    projectedPCA = PCA(n_components = 2).fit_transform(x)
    dfPCA = pd.DataFrame(columns = ["x","y"],data = projectedPCA)
    dfPCA["labelTrue"] = y

    f, (ax1,ax2) = plt.subplots(nrows = 2,ncols = 1, sharex = False)


    sns.scatterplot(x="x",y="y",hue="labelTrue",data=dfPCA.iloc[::frequency],palette="tab10",ax=ax2)
    ax2.set_title("Projection on PCA axes")
    ax2.get_legend().remove()
    ax2.set_xticklabels(['']*len(ax2.get_xticklabels()))
    ax2.set_yticklabels(['']*len(ax2.get_yticklabels()))
    ax2.set_ylabel(None)
    ax2.set_xlabel(None)

    f.set_size_inches(16,16)
    plt.show()

def plotDistribution(x):
    f, axes = plt.subplots(nrows = x.shape[1], ncols=1, sharex = True)

    for i, ax in enumerate(axes):
        sns.kdeplot(x[:, i], ax=ax, shade=True)
    
    f.set_size_inches(16,16)
    plt.show()