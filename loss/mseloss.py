import torch
from torch import nn


model_out = torch.tensor([[1,2,3],
                          [4,5,6]],requires_grad=True,dtype=torch.float32)

y = torch.tensor([[3,4,5],
                  [6,7,8]],dtype=torch.float32)

mse_loss = nn.MSELoss(reduction='mean')
loss = mse_loss(model_out,y)
print(loss)
loss.backward()

