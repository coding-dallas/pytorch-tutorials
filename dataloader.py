import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        super().__init__()
        xy=np.loadtxt('data/wine.csv',delimiter=',',dtype=np.float32, skiprows=1)
        self.x=torch.from_numpy(xy[:,1:])
        self.y=torch.from_numpy(xy[:,[0]])
        self.samples=xy.shape[0]
        
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
dataset=WineDataset()
dataloader=DataLoader(dataset=dataset,batch_size=4,shuffle=True)

dataiter=iter(dataloader)
data=next(dataiter)
features,lables=data
print(features,lables)

#training loop
num_epochs=10
total_samples=len(dataset)
n_iterations=math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (input, output) in enumerate(dataloader):
        
        if (i+1)%10 == 0:
            print('Epoch [{}/{}], Step [{}/{}]'
                              .format(epoch+1, num_epochs, i+1, n_iterations))