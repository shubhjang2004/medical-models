import torch.nn as nn
import torch
from torchvision import models
from transformers import ViTModel, ViTConfig
from collections import OrderedDict

#---------------------------densenetmodel----------------

nih_model=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
in_features=nih_model.classifier.in_features

class Nih_model_head(nn.Module):
    def __init__(self,in_features:int,num_classes:int):
        super().__init__()
        self.head=nn.Sequential(OrderedDict([
            ("f1",nn.Linear(in_features,1024)),
            ("activation",nn.ReLU()),
            ("f2",nn.Linear(1024,256)),
            ("activation",nn.ReLU()),
            ("final_head",nn.Linear(256,15))
        ]))

    def forward(self,x):
        return self.head(x)
        

nih_model.classifier=Nih_model_head(in_features,15)

#------------------------Vit model---------------------





