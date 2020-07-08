import model
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from model import EncoderDecoder
import torch.nn as nn
import torch


def train(data, epochs = 100, batchSize = 32, learningRate = 1e-3):
    data = torch.from_numpy(data).cuda()

    dataset = TensorDataset(data)

    dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

    model = EncoderDecoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)

    for epoch in range(epochs):
        for data in dataLoader:
            data = data[0]
            output = model(data)
            loss = criterion(output, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [%d/%d], loss:%.4f'%(epoch+1, epochs, loss.data.item()))
        if epoch % 2 == 0:
            pic = output.cpu().data
            save_image(pic, 'outputs/%d.png'%(epoch))

    torch.save(model.state_dict(), 'models/encoderDecoder.pth')