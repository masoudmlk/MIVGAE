import torch

from torch.autograd import Variable
a=torch.randn(2, 3)
b=torch.randn(2, 3)

c=torch.cat((a,b),dim=1)
print(c.size())
exit(1)
stacked=torch.stack(lst)
print(stacked)
print(stacked.size())
exit(1)
torch.append(a,b)


print(torch.cat((a, a), 1))
c=torch.stack(a)