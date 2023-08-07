#1)design model
#2)contruct loss and optimizer
#3)training loop
 #forward pass
 #backward pass
 #update weights
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import datasets

#prepare data
x_numpy,y_numpy = datasets.make_regression(n_samples=100,n_features=1, noise=20,random_state=1)

x=torch.from_numpy(x_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))

y=y.view(y.shape[0],1)

n_sampels,n_features=x.shape

#define model
input_size=n_features
output_size=1
model=nn.Linear(input_size,output_size)

#loss and optimizer
loss=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

#training loop
epochs=100

for epoch in range(epochs):
    y_pred=model(x)
    loss_val=loss(y_pred,y)
    
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    
    if epoch%10==0:
        print('epoch:',epoch,'loss:',loss_val.item())
        

#plot
predicted=model(x).detach().numpy()
plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,predicted,'b')
plt.show()
        



 