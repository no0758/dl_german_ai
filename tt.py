import torch
from torch import nn
from torch.autograd import Variable

class ModelSub(nn.Module):
    def __init__(self):
        super(ModelSub, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
        )
        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
        )
    def forward(self, x,weight):
        # x = input[0]
        # weight = input[1]
        x = self.conv1(x)
        weight = self.conv2(weight)
        return x,weight


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        layer = []
        layer.append(ModelSub())
        layer.append(ModelSub())
        self.layer = nn.Sequential(*layer)

    def forward(self,x,weight):
        x,weight= self.layer(x,weight)
        return x, weight


if __name__ == '__main__':
    t2 = ModelSub()
    t = Model()
    x = Variable(torch.rand(3, 1, 28, 28))
    w = Variable(torch.rand(3, 1, 28, 28))
    o = t2(x,w)
    print('=======')
    print(o)
    out = t(x,w)
