import torch
from torch.functional import F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
import numpy as np

#注意力机制 Attention
#公式
#self—attention
#S(Q * k) * v = softmax(Q * k) * v

#Q k V
#A = Q * K
#final_v = softmax(A) * v

Q = torch.randn((8,10,32),dtype=torch.float32)
K = torch.randn((8,10,32),dtype=torch.float32)
V = torch.randn((8,10,32),dtype=torch.float32)


A = torch.matmul(Q,K.permute(0,2,1))

print("A_shape:",A.shape)
attention = F.softmax(A,dim=-1)
print("attention:",attention.shape)
print("V_shape:",V.shape)


#将注意力应用到V上
attention_V = torch.matmul(attention,V)
# plt.figure()
print("attention_V_shape:",attention_V.shape)
x_label = np.linspace(0,10,10)
print("x_label:",x_label.shape)
y_label = attention[0][0]
print("y_label_sum:{}".format(sum(y_label)))
print("y_label_shape:",y_label.shape)
print("y_label:",y_label)
str = ("the", "best", "things","come", "when","you","least","expect","them","to")
# str = ("1", "2", "3","4","5","6","7","8","9","10")
plt.bar(x_label,y_label,0.5,color='coral',label="相关性",tick_label=str)
for a, b in zip(x_label, y_label):
    # print(a,b)
    plt.text(a, b + 0.05, '%.4f' %b, ha='center', fontsize=10)

plt.ylim(0, 1)
plt.title("attention")

plt.legend()
plt.show()























