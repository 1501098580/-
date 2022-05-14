import torch
from torch import nn

input = torch.randn((8,3,32,32)).float()

IN = nn.InstanceNorm2d(3,affine=True) #默认没有学习参数的False
output = IN(input)

print(output[0,0,:,:])


#test
mean = torch.mean(input[0,0,:,:])
var = torch.var(input[0,0,:,:])
# print((input[0,0,:,:] - mean) / torch.pow(var + IN.eps,0.5)* IN.weight[0])
IN_one = ((input[0,0,:,:] - mean) / torch.pow(var + IN.eps,0.5)) * IN.weight[0] + IN.bias[0]
print(IN_one)

print(IN.weight)







