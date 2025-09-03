import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import UNet


model=UNet(
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2, 
    act=("PRELU", {"init": 0.2}),
    norm=("INSTANCE", {"affine": True}),
    dropout=0.0
)
