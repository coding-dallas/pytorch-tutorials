#importing packages

import torch
import torch.nn as nn


#Definging the inputs and outputs
X=torch.tensor([[1],[2],[3],[4],[5]],dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8],[10]],dtype=torch.float32)

X_test=torch.tensor([5],dtype=torch.float32)

n_samples, n_features=X.shape
print(f'N-samples : {n_samples}, N-features : {n_features}')

input_size=n_features
output_size=n_features

#defining the model
class LinearRegression(nn.Module):
    def __init__(self,input_size, ouput_size):
        super(LinearRegression, self).__init__()
        self.linear=nn.Linear(input_size, ouput_size)

    def forward(self, x):
        y_pred=self.linear(x)
        return y_pred
                       
model=LinearRegression(input_size,output_size)

print(f'prediction before training : f(5) = {model(X_test).item():.3f}')

loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.01)

#trainging

epochs=200

for epoch in range(epochs):
    #prediction
    y_pred=model(X)

    #loss
    l=loss(Y,y_pred)
    
    #gradient
    l.backward() #dl/dw - calculate gradient of loss with respect to w
    
    #update weights
    optimizer.step()

    #clear gradient
    optimizer.zero_grad() #clearing the gradient for next iteration
    
    
    if epoch%10==0:
        [w,b]=model.parameters()
        print(f'epoch {epoch+1} : w={w[0][0].item():.3f} , loss={l:.8f}')
        
print(f'prediction after training f(5) = {model(X_test).item():.3f}')
    