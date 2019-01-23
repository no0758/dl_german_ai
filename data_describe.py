import h5py
import numpy as np
import pandas as pd

# train_data = h5py.File('../train/training.h5','r')
# valid_data = h5py.File('../valid/validation.h5','r')
#
# train_sen2 = train_data['sen2']
# valid_sen2 = valid_data['sen2']
#
# train_sen1 = train_data['sen1']
# valid_sen1 = valid_data['sen1']

def get_min(data,channels):

    result = {}

    for i in range(channels):
        channel_i = []
        for j in range(data.shape[0]):
            channel_i.append(data[j,:,:,i].mean())
            if (j+1)%10000 == 0:
                print(i,':',j)
        result[i] = channel_i
    return result

def save(obj,file_path):
    obj.to_csv(file_path, index=None, encoding='utf-8')
#
# train_sen2_result = get_min(train_sen2,10)
# save(pd.DataFrame(train_sen2_result),'../train/train_sen2_mean.csv')
# valid_sen2_result = get_min(valid_sen2,10)
# save(pd.DataFrame(train_sen2_result),'../valid/valid_sen2_mean.csv')
#
# train_sen1_result = get_min(train_sen1,8)
# save(pd.DataFrame(train_sen1_result),'../train/train_sen1_mean.csv')
# valid_sen1_result = get_min(valid_sen1,8)
# save(pd.DataFrame(train_sen1_result),'../valid/valid_sen1_mean.csv')
#
# train_label = {'label':np.argmax(train_data['label'],1)}
# save(pd.DataFrame(train_label),'../train/train_label.csv')
# valid_label = {'label':np.argmax(valid_data['label'],1)}
# save(pd.DataFrame(valid_label),'../valid/valid_label.csv')



# print('combine....')
# train_sen2 = pd.read_csv('../train/train_sen2_mean.csv')
# train_sen1 = pd.read_csv('../train/train_sen1_mean.csv')
# train_label = pd.read_csv('../train/train_label.csv')
#
# train = pd.concat([train_sen1,train_sen2,train_label],axis=1)
#
# save(train,'../train/train.csv')
#
# valid_sen2 = pd.read_csv('../valid/valid_sen2_mean.csv')
# valid_sen1 = pd.read_csv('../valid/valid_sen1_mean.csv')
# valid_label = pd.read_csv('../valid/valid_label.csv')
#
# valid = pd.concat([valid_sen1,valid_sen2,valid_label],axis=1)
# save(valid,'../valid/valid.csv')



print('desc')

def desc(data):
    result = pd.DataFrame()
    for i in range(17):
        desc_i=data[data.label == i].describe()
        desc_i.label = i
        result = result.append(desc_i)
    return result
def rename_col(data,mode):
    cols = data.columns
    cols_ = [i+'_'+mode for i in cols if 'label' !=i ]
    rename = {}
    for i,c in enumerate(cols):
        if c !='label':
            rename[c] = cols_[i]
    data = data.rename(columns=rename)
    return data

train = pd.read_csv('../train/train.csv')
train_desc = desc(train)
train_desc = rename_col(train_desc,'train')
valid = pd.read_csv('../valid/valid.csv')
valid_desc = desc(valid)
valid_desc = rename_col(valid_desc,'valid')

data_desc = pd.concat([train_desc,valid_desc],axis=1)
save(data_desc,'../train/data.csv')


