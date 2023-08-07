import torch 
import torch.nn as nn
import torch.optim as optimizer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split

bc=datasets.load_breast_cancer()
x,y=bc.data,bc.target

n_samples,n_features=x.shape
input_size=n_features

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#scale thefeatures
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


x_train=torch.from_numpy(x_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)


class LogisticRegression(nn.Module):
    def __init__(self,input_size):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(input_size,1)
        
    def forward(self,x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred
model=LogisticRegression(input_size)
criterion=nn.BCELoss()
optimizer=optimizer.SGD(model.parameters(),lr=0.01)

epochs=100

for epoch in range(epochs):
    y_pred=model(x_train)
    loss=criterion(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch%1==0:
        print(f'epoch:{epoch},loss:{loss.item()}')
        
with torch.no_grad():
    y_pred=model(x_test)
    y_pred_cls=y_pred.round()
    accuracy=(y_pred_cls==y_test).float().mean()
    print(f'accuracy:{accuracy},loss:{loss}')