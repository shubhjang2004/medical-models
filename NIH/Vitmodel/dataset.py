import os
import pandas as pd
import numpy as np
import shutil 
from PIL import Image
import torchvision
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split

#----------reading csv file------------
df=pd.read_csv("data\Data_Entry_2017.csv")

#---------getting all disease labels and then mapping them to index---
all_labels=set()
for labels in df["Finding Labels"]:
    for label in labels.split('|'):
        all_labels.add(label.strip())

all_labels=sorted(list(all_labels))
label2idx={label:idx for idx,label in enumerate(all_labels)}


def encode_labels(label_string):
    encoded=np.zeros(len(all_labels),dtype=int)
    for label in label_string.split('|'):
        if label in label2idx:

            encoded[label2idx[label]]=1

    return encoded


df["label_encoded"]=df["Finding Labels"].apply(encode_labels)  

#------------getting all images paths and then craeting a df column of them ------------
all_images=[]
for image_folder in sorted(os.listdir("data")):
    if image_folder.startswith("images"):
        image_folder_path=os.path.join("data",image_folder,"images")
        for image in sorted(os.listdir(image_folder_path)):
            all_images.append(os.path.join(image_folder_path,image))



image_map={os.path.basename(path):path for path in all_images }
df["image_path"]=df["Image Index"].map(image_map)

#------------spltting data in train,test,val-------------
unique_patients=df["Patient ID"].unique()

train_split,test_split=train_test_split(unique_patients,
                                        test_size=0.2,random_state=42)

train_split,val_split=train_test_split(train_split,
                                       test_size=0.2,random_state=42)

train_df=df[df["Patient ID"].isin(train_split)].reset_index(drop=True)
test_df=df[df["Patient ID"].isin(test_split)].reset_index(drop=True)
val_df=df[df["Patient ID"].isin(val_split)].reset_index(drop=True)

#--------------MAking dataset class----------------------

from torchvision.transforms import transforms 
transform =transforms.Compose([        
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]),
])


class NihChestDataset(Dataset):

    def __init__(self,df,transform=None):
        super().__init__()
        
        self.transform=transform
        self.df=df

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):

        img=self.df["image_path"].iloc[idx]
        label=torch.tensor(self.df["label_encoded"].iloc[idx],dtype=torch.float32)
        img=Image.open(img).convert("RGB")
        if self.transform:
            img=self.transform(img)

        return img, label
    



train_dataset=NihChestDataset(train_df,transform)
val_dataset=NihChestDataset(val_df,transform)
test_dataset=NihChestDataset(test_df,transform)




