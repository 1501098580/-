import torch
from torch import nn

model_out = torch.tensor([[1,2,3],
                          [4,5,6]],requires_grad=True,dtype=torch.float32)

#类别
y = torch.tensor(
    [0,2]
)

ce_mean = nn.CrossEntropyLoss(reduction='mean')
loss = ce_mean(model_out,y)
print(loss)
loss.backward()

#softmax
softmax = torch.softmax(model_out,dim=1)
print("q(x):",softmax)
log_model_out = torch.log(softmax)
#-log(q(x))
log_model_out_softmax = -log_model_out
print("-log(q(x))",log_model_out_softmax)
#CrossEntropyLoss
#-∑(p(x)log(q(x)))
#2.4076 + 0.4076
CEL_Sum = log_model_out_softmax[0][0] + log_model_out_softmax[1][2]
CEL_Mean = CEL_Sum / 2
print("crossentropy_mean:",CEL_Mean)
CEL_Mean.backward()









