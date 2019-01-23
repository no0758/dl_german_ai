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
from utils import is_adaptive,is_available,is_parallel,load,Logger
from dataset import H5pyDataset
from trainer import Trainer
from sklearn.model_selection import train_test_split


torch.set_default_tensor_type('torch.FloatTensor')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# config
module_name = 'resnet50'
adaptive = True
train_mode = 'fromscratch'
pretrained_path = '../pretrain/resnet50-19c8e357.pth'
fc_num = 2048
###############
w=96
h=96
batch_size = 10
valid_batch_size = 10
num_workers = 10
num_classes = 17
lr = 0.001
weight_decay = 0.07
num_channels = 3
epoches = 50
class_weights = None


train_data = h5py.File('../train/training.h5','r')
valid_data = h5py.File('../train/training.h5','r')

feature = 'sen2'
label = 'label'

train_transform = None

print('data_loader start......')
from torch.utils.data import dataset
train_dataset = H5pyDataset(train_data,feature,label,train_transform=train_transform,w=w,h=h)
valid_dataset = H5pyDataset(valid_data,feature,label,is_train_set=False,w=w,h=h)
train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size,num_workers=num_workers)
valid_loader = DataLoader(dataset=valid_dataset,shuffle=False,batch_size=valid_batch_size,num_workers=num_workers)
# print('data_loader end......')
#
# print('model load......')
net = getattr(import_module('torchvision.models'), module_name)
model = net(num_classes=num_classes)
is_adaptive(model,fc_num=fc_num,num_classes=num_classes,num_channels = num_channels)
is_available(model)
model = is_parallel(model)
# load(model,train_mode,pretrained_path=pretrained_path)
#
#
#
# # In[28]:
#
# # optimizer
# print('create optimizer......')
optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
optimizer = is_parallel(optimizer)
# loss
print('create loss......')
criterion = nn.CrossEntropyLoss(weight=class_weights)  # nn.MSELoss()
is_available(criterion)
#
#
# # In[29]:
#
# # train
# # pass model, loss, optimizer and dataset to the trainer
# print('='*10)
# print('train......')
# print('='*10)
#
# model.train()
# accuracy = 0
#
# for epoch in range(2):
#     # loop over the dataset multiple times
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for i, data in enumerate(train_loader):
#         # get the inputs
#         inputs, labels = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         # print statistics
#         running_loss += loss.item()
#         if (i+1) % 10 == 0:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#         print((predicted == labels).sum().item()/labels.size(0))
#     print(correct / total)
# print('Finished Training')
#
#
#
# torch.save(model.state_dict(), 'val_model/test.model')

model.load_state_dict(torch.load('val_model/test.model'))
model.eval()

correct = 0
total = 0
running_loss=0
for data in valid_loader:
    images, labels = data
    outputs = model(images)
    loss = criterion(outputs,labels)
    running_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print(running_loss)



# from torchvision import datasets
# train_dataset = datasets.MNIST(root='./data/',
#                                train=True,
#                                transform=transforms.ToTensor(),
#                                download=True)
#
# test_dataset = datasets.MNIST(root='./data/',
#                               train=False,
#                               transform=transforms.ToTensor())
#
# # Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# valid_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
# In[27]:










