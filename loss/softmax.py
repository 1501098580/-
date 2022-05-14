import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'

#softmax
def softmax(input):
    # input = torch.tensor([1,2,3],dtype=torch.float32)
    sum = np.sum([torch.exp(i) for i in input])
    softmax_out = [torch.exp(i)/sum for i in input]
    # print(sum)
    # print(softmax_out)
    # loss = [-torch.log(i) for i in softmax_out]
    # print(loss)
    return softmax_out

input = torch.tensor([1,2,3],dtype=torch.float32)
L = softmax(input)
print(L)

plt.figure()

x_label = torch.tensor([0,1,2]) #总共为三类

input = torch.tensor([1,2,3],dtype=torch.float32)
y_label = softmax(input)
str = ("类别1", "类别2", "类别3")
plt.bar(x_label,y_label,0.5,color='coral',label="类别",tick_label=str)
for a, b in zip(input, y_label):
    print(a,b)
    plt.text(a-1, b + 0.05, '%.4f' %b, ha='center', fontsize=10)

plt.ylim(0, 1)

plt.legend()
plt.show()










