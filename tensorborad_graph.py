

import torch
from tensorboardX import SummaryWriter
from importlib import import_module
from torch.autograd import Variable
import lightcnn

module_name = 'resnet_glr'
num_classes = 17

net = lightcnn.resnet_glr()

dummy_input = Variable(torch.rand(3, 18, 96, 96) )#假设输入13张1*28*28的图片
with SummaryWriter(comment='lightcnn') as w:
    w.add_graph(net,(dummy_input,) )
