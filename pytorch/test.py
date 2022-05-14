from torch import nn
from torchvision import transforms
import torch

import numpy as np
import matplotlib.pyplot as plt

class TestModel(nn.Module):

    pass

if __name__ == '__main__':
    # print('*' * 100)
    # print('查看数据形状')
    # data1 = torch.randn((7,3,32,32))
    # print("shape:",data1.shape)
    # print("size:",data1.size())

    #压缩
    # print('*' * 100)
    # print('压缩')
    # data2 = torch.randn((3,1,32,1,32))
    # print("data2_shape:",data2.size())
    # data2_y = torch.squeeze(data2)
    # print("squeeze:",data2_y.size())
    #
    # data2_y2 = torch.squeeze(data2,dim=0)
    # print("squeeze_0:",data2_y2.size())
    # data2_y3 = torch.squeeze(data2, dim=1)
    # print("squeeze_1:",data2_y3.size())
    # data2_y4 = torch.squeeze(data2_y3, dim=2)
    # print("squeeze_2:",data2_y4.size())

    #增维 解缩
    # print('*' * 100)
    # print("增维")
    # data3 = torch.randn((3,32,32))
    # print("data3_shape",data3.shape)
    # data3_1 = torch.unsqueeze(data3,dim=0)
    # print("unsqueeze_0",data3_1.shape)
    # data3_2 = torch.unsqueeze(data3_1,dim=-1)
    # print("unsqueeze_-1",data3_2.shape)

    #广播机制
    # print('*'*100)
    # print("广播机制")
    # data4 = torch.randn((8,8))
    # print("data4_shape:",data4.size())
    # data5 = torch.randn(1)
    # print("data5_shape",data5.size())
    # data6 = data4 + data5
    # print("(data4+data5)_shape",data6.size())

    #reshape view
    # print('*'*100)
    # #改变形状
    # print("改变形状")
    # x = torch.randn(4, 4)
    # print('x_shape:',x.size())
    # y = x.view(16)
    # print("y_shape:",y.size())
    # z = x.view(-1,8)
    # print("z_shape:",z.size())
    #
    # a = torch.randn(1, 2, 3, 4)
    # print("a_shape:",a.size())
    # b = a.transpose(1,2)
    # print("b_shape:",b.size())
    # c = a.view(1, 3, 2, 4)
    # print("c_shape:",c.size())
    #
    # print("a_b:",torch.equal(a,b))
    # print("b_c:",torch.equal(b,c))
    # print("a_c",torch.equal(a,c))

    #reshape
    data7 = torch.arange(12).reshape(3,4)
    data8 = torch.arange(12).view(3,4)
    print("reshape:",data7)
    print("view:",data8)


    #permute() 列方向展开，行向量变为列向量 和 view()行方向展开,都展开为一维
    # a2 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    # print("a2_shape:",a2.shape)
    # b2 = a2.view(2,3,2)
    # print("b2_shape:",b2.shape)
    # print("b2:",b2)
    #
    # a3 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    # print("a3_shape:",a3.shape)
    # b3 = a3.permute(0,2,1)
    # print("b3_shape:",b3.shape)
    # print("b3:",b3)

    #permute相当于连续使用transpose实现
    # a4 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    # b4 = a4.transpose(1,2)
    # print(b4)

    #torch的连续性 contiguous() 行优先存储为连续
    # a5 = torch.arange(12).reshape(3,4)
    # print("a5:",a5)
    # print("查看a5内存存储布局",[i for i in a5.storage()])
    # a5_y = a5.permute(1,0)
    # print("permute:",a5_y)
    # print("查看a5_permute内存存储布局", [i for i in a5_y.storage()])
    # a5_z =a5_y.is_contiguous()
    # print("permute()存储是否连续:",a5_z)
    # a5_f =a5_y.contiguous()
    # a5_g = a5_f.is_contiguous()
    # print("contiguous()存储是否连续:",a5_g)
    # print("查看a5_permute_contiguous内存存储布局", [i for i in a5_f.storage()])
    # a5_h = a5_f.view(3,4)
    # print("view:",a5_h)
    # a5_l = a5_h.is_contiguous()
    # print("view()存储是否连续:",a5_l)
    # print("查看a5_permute_contiguous_view内存存储布局", [i for i in a5_h.storage()])
    #总结：
    # 1、reshape = contiguous + view
    # 2、permute = transpose().transpose()......
    # 3、一般需要将tensor拉开时使用view()，而在需要转置时使用permute()。











