import torch
from torch.autograd import Variable


a = torch.LongTensor([1,2,3])

b = Variable(torch.LongTensor(a))
print(a)

print(b)