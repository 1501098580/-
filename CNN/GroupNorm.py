import torch
from torch import nn

input = torch.randn((8,6,32,32))

GN = nn.GroupNorm(3,6) #Separate 6 channels into 3 groups

output = GN(input)
print(output[0,0,:,:])

#手动
input_ = input.reshape(8,3,2,32,32)
print(input_.shape)

mean = torch.mean(input_[0,0,:,:,:])
var = torch.var(input_[0,0,:,:,:])

GN_one = (input_[0,0,:,:,:] - mean) / torch.pow(var + GN.eps,0.5) * GN.weight[0] + GN.bias[0]
print(GN_one[0])
print(GN.weight)