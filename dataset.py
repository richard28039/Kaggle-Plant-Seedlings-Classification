import os
import pandas as pd
from PIL import Image
from utils import *

from torch.utils.data import Dataset

train_path = 'dataset/train/'


def get_train_data():
    clas = os.listdir(train_path)
    imgs = []
    labels = []
    for i in clas:
        for j in range(len(os.listdir(train_path+ i))):
            imgs.append(train_path + i + '/' + os.listdir(train_path+ i)[j])
            labels.append(i)
            
    df = pd.DataFrame({"imgs" : imgs, "labels" : labels})

    return df

class train_valid_dataset(Dataset):
    def __init__(self, data, tfm=None):
#         print(file_path)
        self.data = data
        self.transform = tfm
       
        self.imgs = list(self.data["imgs"])
        self.labels = list(self.data["labels"])

#         print(len(self.imgs)==len(self.labels)) 
        
    def __getitem__(self,idx):
        img_as_img = Image.open(self.imgs[idx]).convert('RGB')
        img_as_img = self.transform(img_as_img)

        label = self.labels[idx]
        class_to_n, _ = utils_function.class_to_num()
        label = class_to_n[label]
        
        return img_as_img,label
    
    def __len__(self):   
        return len(self.imgs)


class test_dataset(Dataset):
    def __init__(self, file_path, tfm=None):
        self.file_path = file_path
        self.transform = tfm
        self.imgs = []
        for i in range(len(os.listdir(self.file_path))):
            self.imgs.append(self.file_path + os.listdir(self.file_path)[i])
    
    def __getitem__(self, idx):
        img_as_img = Image.open(self.imgs[idx]).convert('RGB')
        img_as_img = self.transform(img_as_img)

        return img_as_img, self.imgs[idx]

    def __len__(self):

        return len(self.imgs)
