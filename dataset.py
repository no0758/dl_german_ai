from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

# create data_set
class H5pyDataset(Dataset):
    def __init__(self, data, feature, label, train_transform=None,
                 target_transform=None, is_train_set=True,w = 224,h=224):
        self.data = data
        self.sen2 = self.data[feature]
        self.sen1 = self.data['sen1']
        self.label = np.argmax(self.data[label], axis=1)
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.train_transform = train_transform
        self.target_transform = target_transform
        self.is_train_set = is_train_set
        self.len = self.sen2.shape[0]
        self.weight = w
        self.height = h

    def __getitem__(self, index):

        sen1 =self.sen1.__getitem__(index).astype('float32')

        label = self.label.__getitem__(index)
        sen2 = self.sen2.__getitem__(index).astype('float32')
        image_sen2 = np.zeros([ self.weight,self.height,10]).astype('float32')
        image_sen1 = np.zeros([ self.weight,self.height,8]).astype('float32')
        for i in range(10):
            image_i = Image.fromarray(sen2[:, :, i])
            image_i = image_i.resize((self.weight,self.height))
            image_i = np.array(image_i)
            image_sen2[:,:,i] = image_i
        for i in range(8):
            image_i = Image.fromarray(sen1[:, :, i])
            image_i = image_i.resize((self.weight, self.height))
            image_i = np.array(image_i)
            image_sen1[:, :, i] = image_i
        image = np.concatenate((image_sen1,image_sen2),axis=2).astype('float32')
        image = self.transform(image)
        if self.is_train_set and self.train_transform is not None:
            image = self.train_transform(image)

        return image, label

    def __len__(self):
        return self.len

    def get_len(self):
        return self.len

class H5pyDatasetTV(Dataset):
    def __init__(self, data, data_t,feature, label, train_transform=None,
                 target_transform=None, is_train_set=True,w = 224,h=224):
        self.data = data
        self.data_t = data_t
        self.sen2 = self.data[feature]
        self.sen1 = self.data['sen1']
        self.label = np.argmax(self.data[label], axis=1)

        self.sen2_t = self.data_t[feature]
        self.sen1_t = self.data_t['sen1']
        self.label_t = np.argmax(self.data_t[label], axis=1)

        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.train_transform = train_transform
        self.target_transform = target_transform
        self.is_train_set = is_train_set
        self.len = self.sen2.shape[0]
        self.len_t = self.sen2_t.shape[0]
        self.weight = w
        self.height = h

    def __getitem__(self, index):

        if index < self.len:
            sen1 = self.sen1.__getitem__(index).astype('float32')
            label = self.label.__getitem__(index)
            sen2 = self.sen2.__getitem__(index).astype('float32')
        else:
            sen1 = self.sen1_t.__getitem__(index-self.len).astype('float32')
            label = self.label_t.__getitem__(index-self.len)
            sen2 = self.sen2_t.__getitem__(index-self.len).astype('float32')

        image_sen2 = np.zeros([ self.weight,self.height,10]).astype('float32')
        image_sen1 = np.zeros([ self.weight,self.height,8]).astype('float32')
        for i in range(10):
            image_i = Image.fromarray(sen2[:, :, i])
            image_i = image_i.resize((self.weight,self.height))
            image_i = np.array(image_i)
            image_sen2[:,:,i] = image_i
        for i in range(8):
            image_i = Image.fromarray(sen1[:, :, i])
            image_i = image_i.resize((self.weight, self.height))
            image_i = np.array(image_i)
            image_sen1[:, :, i] = image_i
        image = np.concatenate((image_sen1,image_sen2),axis=2).astype('float32')
        image = self.transform(image)
        if self.is_train_set and self.train_transform is not None:
            image = self.train_transform(image)

        return image, label

    def __len__(self):
        return self.len + self.len_t

    def get_len(self):
        return self.len + self.len_t


class H5pyDatasetTest(Dataset):
    def __init__(self, data, feature,label=None, train_transform=None,
                 target_transform=None, is_train_set=True,w = 224,h=224):
        self.data = data
        self.sen2 = self.data[feature]
        self.sen1 = self.data['sen1']
        self.label = np.argmax(self.data[label], axis=1) if label!=None else None
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.train_transform = train_transform
        self.target_transform = target_transform
        self.is_train_set = is_train_set
        self.len = self.sen2.shape[0]
        self.weight = w
        self.height = h

    def __getitem__(self, index):

        sen1 =self.sen1.__getitem__(index).astype('float32')

        if type(self.label)!= type(None):
            label = self.label.__getitem__(index)
        else:
            label = 0
        sen2 = self.sen2.__getitem__(index).astype('float32')
        image_sen2 = np.zeros([ self.weight,self.height,10]).astype('float32')
        image_sen1 = np.zeros([ self.weight,self.height,8]).astype('float32')
        for i in range(10):
            image_i = Image.fromarray(sen2[:, :, i])
            image_i = image_i.resize((self.weight,self.height))
            image_i = np.array(image_i)
            image_sen2[:,:,i] = image_i
        for i in range(8):
            image_i = Image.fromarray(sen1[:, :, i])
            image_i = image_i.resize((self.weight, self.height))
            image_i = np.array(image_i)
            image_sen1[:, :, i] = image_i
        image = np.concatenate((image_sen1,image_sen2),axis=2).astype('float32')
        image = self.transform(image)
        if self.is_train_set and self.train_transform is not None:
            image = self.train_transform(image)

        return image, label

    def __len__(self):
        return self.len

    def get_len(self):
        return self.len