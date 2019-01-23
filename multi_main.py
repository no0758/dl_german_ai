
# coding: utf-8

# In[21]:

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import flip
from torch.utils.data import Dataset, DataLoader
from torch.utils import trainer
from torch.utils.trainer import plugins
from torchvision import transforms
from importlib import import_module
from torch import nn
import os
from utils import is_adaptive,is_available,is_parallel,load,Logger,channels_conv
from muti_dataset import H5pyDataset
from multi_trainer import Trainer
from sklearn.model_selection import train_test_split



torch.set_default_tensor_type('torch.FloatTensor')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1,4'


# In[22]:

# config
module_name = 'resnet50'
adaptive = True
train_mode = 'fromscratch'
pretrained_path = '../pretrain/resnet50-19c8e357.pth'
# pretrained_path = 'val_model/eval_best_sen12.model'

fc_num = 2048
###############
w=96
h=96
batch_size = 64
valid_batch_size = 64
num_workers = 10
task1_classes = 3
num_classes = 17
lr = 0.001
weight_decay = 1e-5
num_channels = 10
epoches = 200

# weights = [1.438277, 6.933416, 8.994341, 2.455118,  4.680645, 10.015155, 0.927729,  11.160555,  3.855082, 3.392495, 12.175409,2.700033,2.600989,11.742620,0.678840,2.241419,14.007878]
# weights = [94.21484375,
#  19.23365231259968,
#  10.25031874203145,
#  28.40871613663133,
#  31.86129458388375,
#  12.654249737670515,
#  50.88396624472574,
#  7.104270986745213,
#  12.60135841170324,
#  28.0453488372093,
#  10.54613030170529,
#  63.138743455497384,
#  20.065723793677204,
#  8.780123771386968,
#  119.4009900990099,
#  35.891369047619044,
#  9.244538137217324]
# class_weights = torch.FloatTensor(weights)
# is_available(class_weights)

class_weights = None
# In[23]:

train_data = h5py.File('../train/training.h5','r')
valid_data = h5py.File('../valid/validation.h5','r')

# data = h5py.File('../valid/validation.h5','r')
# data_set = {'sen2':np.array(data['sen2']),'label':np.array(data['label'])}
# train_data = h5py.File('../train/train_data.h5','w')
# valid_data = h5py.File('../valid/valid_data.h5','w')
# X_train, X_test, y_train, y_test = train_test_split(data_set['sen2'], data_set['label'], test_size=0.2, random_state=0)
# train_data['sen2'] = X_train
# train_data['label'] = y_train
# valid_data['sen2'] = X_test
# valid_data['label'] = y_test



feature = 'sen2'
label = 'label'


# In[24]:

# transform
mean = [0.1237569611768191, 0.1092774636368305, 0.10108552032678805, 0.11423986161140066, 0.15926566920230753,
             0.18147236008771511, 0.1745740312291362, 0.19501607349634489, 0.15428468872076573, 0.10905050699570018]
std = [0.032418274013566131, 0.040232448183413781, 0.057602887097067637, 0.05693729692765731, 0.069060142757838186,
            0.081302909240360877, 0.080020583972139958, 0.091254124441033077, 0.092223528280801242,
            0.080112723816366818]
# train_transform  = transforms.Compose([
#                             transforms.Normalize(mean = mean,std = std)])
train_transform = None

# In[25]:

# create data_set(band 2 3 4) data_loader
print('data_loader start......')
train_dataset = H5pyDataset(train_data,feature,label,train_transform=train_transform,w=w,h=h)
valid_dataset = H5pyDataset(valid_data,feature,label,is_train_set=False,w=w,h=h)
train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size,num_workers=num_workers)
valid_loader = DataLoader(dataset=valid_dataset,shuffle=False,batch_size=valid_batch_size,num_workers=num_workers)
print('data_loader end......')



# model
print('model load......')
net1 = getattr(import_module('torchvision.models'), module_name)
net2 = getattr(import_module('torchvision.models'), module_name)
model1 = net1(num_classes=task1_classes)
model2 = net2(num_classes=num_classes)
net = getattr(import_module('multi_model'), 'multi_model')
model = net(model1,model2, task1_classes=task1_classes,task2_classes=num_classes,channels = num_channels)
load(model,train_mode,pretrained_path=pretrained_path)



# In[28]:

# optimizer
print('create optimizer......')
optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
is_parallel(optimizer)
# loss
print('create loss......')
criterion = nn.CrossEntropyLoss(weight=class_weights)  # nn.MSELoss()
is_available(criterion)


# In[29]:

# train
# pass model, loss, optimizer and dataset to the trainer
print('='*10)
print('train......')
print('='*10)
e = Trainer(model, criterion, optimizer, dataset=train_loader,valid_dataset=valid_loader,
            file_path='multi_logger.log',save_path='multi_val_model')
# register some monitoring plugins
e.register_plugin(plugins.ProgressMonitor())
e.register_plugin(plugins.AccuracyMonitor())
e.register_plugin(plugins.LossMonitor())
e.register_plugin(plugins.TimeMonitor())
e.register_plugin(Logger(['progress', 'accuracy', 'loss', 'time'],file_path='multi_logger.log'))
e.run(epoches)


