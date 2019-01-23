
# coding: utf-8

# In[21]:

import h5py
import torch
from torch.utils.data import  DataLoader
from torch.utils.trainer import plugins
from importlib import import_module
import os
from utils import is_adaptive,is_available,is_parallel,load,Logger,channels_conv
from dataset import H5pyDataset,H5pyDatasetTV
from trainer import Trainer
from focal_loss import FocalLoss

torch.set_default_tensor_type('torch.FloatTensor')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


# In[22]:

# config
module_name = 'resnet_glr'
adaptive = True
train_mode = 'finetune'
# pretrained_path = '/Users/lidandan/Downloads/LightCNN_29Layers_checkpoint.pth'
# pretrained_path = 'val_model_lightcnn_valid/eval_best.model'

fc_num = 2048
###############
w=96
h=96
batch_size = 64
valid_batch_size = 64
num_workers = 10
num_classes = 17
lr = 0.0001
weight_decay = 1e-5
num_channels = 10
epoches = 20

# weights = [1.438277, 6.933416, 8.994341, 2.455118,  4.680645, 10.015155, 0.927729,  11.160555,  3.855082, 3.392495, 12.175409,2.700033,2.600989,11.742620,0.678840,2.241419,14.007878]
# train+valid
weights = [1.420992, 6.863661, 9.015326, 2.509418,  4.615314, 9.914719, 0.970890,
           11.286964,  4.046592, 3.394728, 12.051488, 2.647536, 2.715512, 11.728671,
           0.682737, 2.266492,13.868961]


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

# train_data = h5py.File('../train/training.h5','r')
# valid_data = h5py.File('../valid/validation.h5','r')


"""
整合train和valid
"""

print('==================train+valid=========')
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
train_data = h5py.File('../train/training.h5','r')
valid_t = pickle.load(open('../valid/v_train.pkl','rb'))
valid_data = pickle.load(open('../valid/v_valid.pkl','rb'))


"""
整合train和valid
"""

feature = 'sen2'
label = 'label'

# transform
# mean = [0.1237569611768191, 0.1092774636368305, 0.10108552032678805, 0.11423986161140066, 0.15926566920230753,
#              0.18147236008771511, 0.1745740312291362, 0.19501607349634489, 0.15428468872076573, 0.10905050699570018]
# std = [0.032418274013566131, 0.040232448183413781, 0.057602887097067637, 0.05693729692765731, 0.069060142757838186,
#             0.081302909240360877, 0.080020583972139958, 0.091254124441033077, 0.092223528280801242,
#             0.080112723816366818]
# train_transform  = transforms.Compose([
#                             transforms.Normalize(mean = mean,std = std)])
train_transform = None

# In[25]:

# create data_set(band 2 3 4) data_loader
print('data_loader start......')
print('train_valid')
train_dataset = H5pyDatasetTV(train_data,valid_t,feature,label,train_transform=train_transform,w=w,h=h)
# train_dataset = H5pyDataset(train_data,feature,label,train_transform=train_transform,w=w,h=h)
valid_dataset = H5pyDataset(valid_data,feature,label,is_train_set=False,w=w,h=h)
train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size,num_workers=num_workers)
valid_loader = DataLoader(dataset=valid_dataset,shuffle=False,batch_size=valid_batch_size,num_workers=num_workers)
print('data_loader end......')



# model
print('model load......')
# net = getattr(import_module('torchvision.models'), module_name)
net = getattr(import_module('lightcnn'), module_name)

model = net(num_classes=num_classes,channels = 18)
is_available(model)
# is_adaptive(model,fc_num=fc_num,num_classes=num_classes,num_channels = num_channels)
# channels_conv(model,fc_num=fc_num,num_classes=num_classes)
# load(model,train_mode,pretrained_path=pretrained_path)



# In[28]:

# optimizer
print('create optimizer......')
optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
is_parallel(optimizer)
# loss
print('create loss......')
# criterion = nn.CrossEntropyLoss(weight=class_weights)  # nn.MSELoss()
criterion = FocalLoss(class_num=num_classes,alpha=class_weights, gamma=2)


# In[29]:

# train
# pass model, loss, optimizer and dataset to the trainer
print('='*10)
print('train......')
print('='*10)
e = Trainer(model, criterion, optimizer, dataset=train_loader,valid_dataset=valid_loader,
            file_path='logger_lightcnn_valid_focalloss_.log',save_path='val_model_lightcnn_valid_focalloss')
# register some monitoring plugins
e.register_plugin(plugins.ProgressMonitor())
e.register_plugin(plugins.AccuracyMonitor())
e.register_plugin(plugins.LossMonitor())
e.register_plugin(plugins.TimeMonitor())
e.register_plugin(Logger(['progress', 'accuracy', 'loss', 'time'],file_path='logger_lightcnn_valid_focalloss_.log'))
e.run(epoches)

