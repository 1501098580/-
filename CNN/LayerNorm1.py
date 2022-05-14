import torch
from torch import nn

input = torch.randn((8,3,32,32))

LN = nn.LayerNorm((3,32,32))
output = LN(input)
print(output[0])

#test
mean = torch.mean(input[0,:,:,:])
var = torch.var(input[0,:,:,:])
print(mean)
print(var)

LN_one = ((input[0] - mean) / torch.pow(var + LN.eps,0.5)) * LN.weight[0] + LN.bias[0]
print(LN_one)

print(LN.weight.shape)


