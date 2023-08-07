#coding linear regression from scratch using the numpy

import torch

#expresssion
#f=w*x
#Here w=2
#f=2*x

#Definging the inputs and outputs
X=torch.tensor([1,2,3,4,5],dtype=torch.float32)
Y=torch.tensor([2,4,6,8,10],dtype=torch.float32)

w=torch.tensor(0,dtype=torch.float32, requires_grad=True)

#model prediction
def forward(x):
    return w*x

#loss=MSE
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()

print(f'prediction before training : f(5) = {forward(5):.3f}')

#trainging
lr=0.01
epochs=100
print("W",w,type(w))
for epoch in range(epochs):
    #predication
    y_pred=forward(X)

    #loss
    l=loss(Y,y_pred)
    
    #gradient
    l.backward() #dl/dw - calculate gradient of loss with respect to w
    
    #update weights
    with torch.no_grad(): #writing to not to affect the updation of weights by gradient calculation
        w -=(lr*(w.grad))
    
    #clear gradient
    w.grad.zero_() #clearing the gradient for next iteration
    
    
    if epoch%10==0:
        print(f'epoch {epoch+1} : w={w:.3f} , loss={l:.8f}')
        
print(f'prediction after training f(5) = {forward(5):.3f}')
    