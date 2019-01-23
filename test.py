
# coding: utf-8

# In[21]:

import h5py
import torch
from torch.utils.data import DataLoader

from importlib import import_module
from torch import nn
import os
from utils import is_available,is_parallel,load
from dataset import H5pyDatasetTest
from trainer import Trainer

torch.set_default_tensor_type('torch.FloatTensor')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


# In[22]:

# config
module_name = 'resnet_glr'
adaptive = True
train_mode = 'finetune'
pretrained_path = 'val_model_lightcnn_valid_focalloss/eval_best.model'
fc_num = 2048
###############
w=96
h=96
batch_size = 64
valid_batch_size = 64
num_workers = 10
num_classes = 17
lr = 0.001
weight_decay = 0.07
num_channels = 10
epoches = 50

class_weights = None
valid_data = h5py.File('../testa/round1_test_a_20181109.h5','r')


feature = 'sen2'
# label = 'label'
label=None
print('data_loader start......')
valid_dataset = H5pyDatasetTest(valid_data,feature,label,is_train_set=False,w=w,h=h)
valid_loader = DataLoader(dataset=valid_dataset,shuffle=False,batch_size=valid_batch_size,num_workers=num_workers)
print('data_loader end......')

print('model load......')
net = getattr(import_module('lightcnn'), module_name)
model = net(num_classes=num_classes,channels = 18)
load(model,train_mode,pretrained_path=pretrained_path)
is_available(model)


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
print('test......')
print('='*10)
e = Trainer(model, criterion, optimizer, dataset=None,valid_dataset=valid_loader,
            file_path='logger_rgb.log',save_path='val_model_rgb')
result = e.test_and_save()
import pickle
pickle.dump(result,open('result/testa_lightcnn_valid_focalloss.result','wb'))


