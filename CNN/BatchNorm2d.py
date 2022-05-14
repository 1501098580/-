import torch
from torch import nn


#BatchNorm2d
input = torch.randn((8,32,32,3)).float()
input_ = input.permute(0,3,1,2)
print(input_.shape)

BN = nn.BatchNorm2d(3)
# batch_relu = nn.ReLU(batch)
output = BN(input_)
# print(output[:,0,:,:])

#test
X = input_[:,0,:,:]  #N,H,W
# print(X.shape)

mean = torch.mean(X)
var = torch.var(X)
print(BN.weight)
BN_one = (input_[:,0,:,:] - mean) / torch.pow(var + BN.eps,0.5) * BN.weight[0] + BN.bias[0]

# print("BN_one:",BN_one)

