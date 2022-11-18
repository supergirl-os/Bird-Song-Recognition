import torch.utils.data as torchdata
import torch
import torch.functional as F
import numpy as np
import os
import h5py
class Bird_train_Dataset(torchdata.Dataset):
    def get_path(self,file_path):
        files = os.listdir(file_path)
        path_file = []
        for i in files:
            path_file.append(file_path + os.sep + i)
            # print(file_path+os.sep+i)
        return path_file
    def preprocess_audio(self,path):
        # print(path)
        tmp = h5py.File(path, 'r')
        train_set_data = tmp['audio_data'][:]
        train_set_label = tmp['audio_label'][:]
        data = np.array(train_set_data, dtype="float64")
        label = np.array(train_set_label, dtype="int")
        # print(path,len(path),label,"\n")
        return data,label

    def __init__(self, file_path="/media/brl/Data1/gly/birds_mel",sub="train_data_new",transform=None):
        self.data_files = self.get_path(file_path+os.sep+sub)
        self.transform=transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        x,label = self.preprocess_audio(self.data_files[index])
        # x = np.pad(x,(1,7),'constant')
        x = x/170


        # print(x.size(),x)
        # print(type(x))
        x = x.reshape(1,x.shape[0],x.shape[1])
        x = torch.from_numpy(x)
        # print("data",x.shape)
        label = torch.from_numpy(label)
        x = x.type(torch.FloatTensor)
        # t_max = torch.max(x,)[0]
        # x = x/t_max
        # label = label.type(torch.FloatTensor)
        # print("data:", x[0], "label", label)
        #if self.transform:
        # x=self.transform(x)

        return x,label


class Bird_dev_Dataset(torchdata.Dataset):
    def get_path(self,file_path):
        files = os.listdir(file_path)
        path_file =[]
        for i in files:
            path_file.append(file_path+os.sep+i)
            # print(file_path+os.sep+i)
        return path_file
    def __init__(self, file_path="/media/brl/Data1/gly/birds_mel",sub="dev_data_new",transform = None):
        self.data_files = self.get_path(file_path+os.sep+sub)
        self.transform = transform
    def preprocess_audio(self,path):

        tmp = h5py.File(path, 'r')
        train_set_data = tmp['audio_data'][:]
        train_set_label = tmp['audio_label'][:]
        data = np.array(train_set_data, dtype="float64")
        label = np.array(train_set_label, dtype="int")
        return data,label
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, item):
        x,label = self.preprocess_audio(self.data_files[item])
        x = x/170
        x = x.reshape(1,x.shape[0],x.shape[1])
        x = torch.from_numpy(x)
        x = x.type(torch.FloatTensor)
        label = torch.from_numpy(label)
        #if self.transform:
        # x = self.transform(x)
        #print("x=", x)
        #print("label=", label)
        return x,label

class Bird_test_Dataset(torchdata.Dataset):
    def get_path(self,file_path):
        files = os.listdir(file_path)
        print("files",len(files))
        return files
    def preprocess_audio(self,path):
        # data = []
        #for file in path:
        #print("file=",path)
        tmp = h5py.File("/media/brl/Data1/gly/birds_mel/test_data_new/"+path, 'r')
        #print("successfully!")
        train_set_data = tmp['audio_data'][:]
        # data.append(train_set_data)
        data = np.array(train_set_data,dtype="float64")
        # print("data=",type(data))
        return data,path
    def __init__(self, file_path="/media/brl/Data1/gly/birds_mel",sub="test_data_new",transform=None):
        self.data_files = self.get_path(file_path + os.sep + sub)
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, item):
        #print("path=",self.data_files[item])
        # print("item",item)
        x,name= self.preprocess_audio(self.data_files[item])
        x = x / 170

        # print(type(x),x.shape)
        #print(x.shape[1])
        x = x.reshape(1, x.shape[0], x.shape[1])
        x = torch.from_numpy(x)
        # x = x.view(1,x.size(0),  x.size(1))绝了为啥这里还要view？前面已经有了reshape两个功能一样，只是针对的数据类型不同
        x = x.type(torch.FloatTensor)
        # x = self.transform(x)
        # print("x=",x)
        name=name.split('_')[1][4:]
        return x,name


# Bird_train_Dataset()
# Bird_dev_Dataset()


