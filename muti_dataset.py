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
        # ndvi = (self.data[feature][:,:,:,6] - self.data[feature][:,:,:,2])/(self.data[feature][:,:,:,6] + self.data[feature][:,:,:,2]) #6-2/6+2
        # print('# ndvi')
        # self.sen2 = self.data[feature][:, :, :, 0:3]  # use band2，band3，band4 ---bgr
        # self.sen2 = self.data[feature][:, :, :, [1, 2, 6]]  # use band8，band3，band4 ---CI
#         self.sen2 = self.data[feature][:, :, :, [2, 7, 9]] # use band8a，band12，band4 ---swir
        self.sen2 = self.data[feature]
        self.sen1 = self.data['sen1']
        # self.sen2 = np.stack((self.data[feature][:, :, :, 5],ndvi,self.data[feature][:, :, :,9]),axis=3)
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

        sen1 =self.sen1.__getitem__(index)

        label = self.label.__getitem__(index)
        # label = label if label in [16] else 0
        if label==16:
            label1 = 2
        elif label in [13,12,11,10]:
            label1 = 1
        else:
            label1 = 0
        # label = label if label not in  [15,14,13,12,11,10] else 0
        image_tmp= self.sen2.__getitem__(index).astype('float32')
        # image= self.sen2.__getitem__(index)[:, :, :3].astype('float32')
        image = np.zeros([ self.weight,self.height,10]).astype('float32')

        for i in range(10):
            image_i = Image.fromarray(image_tmp[:, :, i])
            image_i = image_i.resize((self.weight,self.height))
            image_i = np.array(image_i)
            image[:,:,i] = image_i
        # image = image[ :, :, 1:4]
        # image = np.zeros([ self.weight,self.height,11]).astype('float32')
        # image[ :32, :32,:10] = image_tmp
        # image[:32, :32, 10] = (image_tmp[:,:,6] - image_tmp[:,:,2])/(image_tmp[:,:,6] + image_tmp[:,:,2])
        # image[ :32, 32:64, :] = image_tmp[ :, :, 3:6]
        # image[ 32:64, :32, :] = image_tmp[ :, :, 6:9]
        # image[ 32:64, 32:64, 1] = image_tmp[ :, :, 9]


        # image = np.concatenate((image_tmp,sen1),axis=2).astype('float32')
        image = self.transform(image)
        if self.is_train_set and self.train_transform is not None:
            image = self.train_transform(image)

        return image, label1, label

    def __len__(self):
        return self.len

    def get_len(self):
        return self.len