import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self,img_txt_file,img_dir,transform=None,target_transform=None,
                 loader=default_loader):
        img_list=[]

        with open(img_txt_file,'r') as f:
            for line in f:
                line.strip('\n')
                img_path,label=line.split()
                img_list.append((img_path,label))

        self.img_list=img_list
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader

    def __getitem__(self, index):
        img_path,label=self.img_list[index]

        img_path=os.path.join(self.img_dir,img_path)

        img=self.loader(img_path)

        if self.transform is not None:
            img=self.transform(img)
        else:
            img=torch.Tensor.from_numpy(img)

        return img,label

    def __len__(self):
        return len(self.img_list)