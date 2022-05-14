import torch
from torch import nn

#还有一种LayerNorm，是对最后一维进行归一化
input  = torch.randn((8,3,32,32))
input_ = input.permute(0,2,3,1)
# print(input_)
print(input_.shape)

LN = nn.LayerNorm(3)
output = LN(input_)
print(output[0,0,0,:])

#test
mean = torch.mean(input_[0,0,0,:])
var = torch.var(input_[0,0,0,:],unbiased=False) #注意，这里使用的是，有偏样本方差
# print(LN.weight.shape)
LN_one = ((input_[0,0,0,:] - mean) / torch.pow(var + LN.eps,0.5)) * LN.weight[0] + LN.bias[0]
print(LN_one)


