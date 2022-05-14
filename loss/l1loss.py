import torch
from torch import nn


model_x = torch.tensor([[1,2,3],
                        [4,5,6]],requires_grad=True,dtype=torch.float32)

model_y = torch.tensor([[3,4,5],
                        [6,7,8]],dtype=torch.float32)

l1_none = torch.nn.L1Loss(reduction='none')
loss = l1_none(model_x,model_y)
print(loss)

l1_sum = torch.nn.L1Loss(reduction='sum')
loss = l1_sum(model_x,model_y)
print(loss)
loss.backward()

l1_mean = torch.nn.L1Loss(reduction='mean')
loss = l1_mean(model_x,model_y)
print(loss)
loss.backward()





