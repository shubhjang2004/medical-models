import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import UNet

class Convolution3D(nn.Module):
    def __init__(self,in_ch,out_ch,transpose=False,stride=1,k=3,pad=1,out_pad=0):
        super().__init__()
        if not transpose:
            self.conv=nn.Conv3d(in_ch,out_ch,kernel_size=k,stride=stride,padding=pad,bias=True)
        else:
            self.conv=nn.Conv3d(in_ch,out_ch,kernel_size=k,stride=stride,padding=pad,bias=True)

        self.norm=nn.InstanceNorm3d(out_ch,affine=False,track_running_stats=False)
        self.drop=nn.Dropout(p=0.0,inplace=False)        
        self.act=nn.PReLU(num_parameters=1)

    def forward(self,x):
        x=self.conv(x)
        x=self.norm(x)
        x=self.norm(x)
        x=self.drop(x)
        x=self.act(x)
        return x


class ResidualUnit3D(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1,use_1x1_if_needed=True):
        super().__init__()

        self.unit0=Convolution3D(in_ch,out_ch,transpose=False,stride=stride,k=3,p=1)
        self.unit1=Convolution3D(in_ch,out_ch,transpose=False,stride=1,k=3,p=1)

        if stride !=1:
            self.residual=nn.Conv3d(in_ch,out_ch,kernel_size=3,stride=stride,padding=1,bias=True)

        elif in_ch!= out_ch and use_1x1_if_needed:
            self.residual=nn.Conved(in_ch,out_ch,kernel_size=1,stride=1,padding=0,bias=True)

        else:
            self.residual=nn.Identity()

    def forward(self,x):
        y=self.unit0(x)
        y=self.unit1(y)
        r=self.residual(x)           

        return y+r



class ResUNet3D_Exact(nn.Module):
    def __init__(self,in_channels=1,out_channels=2):
        super().__init__()

        # ---------- Encoder ----------
        # Stage 1: 1 -> 16 (down by 2)
        self.enc1=ResidualUnit3D(in_channels,16,stride=2)
        # Stage 2: 16 -> 32 (down by 2)
        self.enc2=ResidualUnit3D(16,32,stride=2)

        #stage 3:32->64 down(by 2)
        self.enc3=ResidualUnit3D(32,64,stride=2)

        #stage 4  64->128

        self.enc4=ResidualUnit3D(64,128,stride=2)

        self.bottleneck=ResidualUnit3D(128,256,stride=1,use_1x1_if_needed=True)



        # ---------- Decoder (concat FIRST, then upsample) ----------
        # Up 1: cat(enc4[128], bottleneck[256]) = 384 -> upconv to 64 -> RU(64->64)
        self.up1=Convolution3D(384,64,transpose=True,stride=2,k=3,p=1,out_pad=1)
        self.up1_refine=ResidualUnit3D(64,64,stride=1)

        # Up 2: cat(enc3[64], up1[64]) = 128 -> upconv to 32 -> RU(32->32)
        self.up2=Convolution3D(128,32,transpose=True,stride=2,k=3,pad=1,out_pad=1)
        self.up2_refine=ResidualUnit3D(32,32,stride=1)

         # Up 3: cat(enc2[32], up2[32]) = 64 -> upconv to 16 -> RU(16->16)
        self.up3=Convolution3D(64,16,transpose=True,stride=2,k=3,p=1,out_pad=1)
        self.up3_refine=ResidualUnit3D(16,16,stride=1)

        ## Up 4 (final): cat(enc1[16], up3[16]) = 32 -> upconv to 2 -> RU(2->2)

        self.up4=Convolution3D(32,out_channels,transpose=True,stride=2,k=3,p=1,out_pad=1)
        self.up4_refine=ResidualUnit3D(out_channels,out_channels,stride=1)

    def forward(self,x):

        # ----- Encoder -----
        s1=self.enc1(x)   # (B, 16, D/2,  H/2,  W/2)
        s2=self.enc2(s1)  # (B, 32, D/4,  H/4,  W/4)
        s3=self.enc3(s2)  # (B, 64, D/8,  H/8,  W/8)
        s4=self.enc4(s3)  # (B,128, D/16, H/16, W/16)

        b=self.bottleneck(s4) # (B,256, D/16, H/16, W/16)
       
         # ----- Decoder (concat -> upconv -> RU) -----
        x=torch.cat([s4,b],dim=1)  # 128 + 256 = 384
        x=self.up1(x)              # -> (B, 64, D/8,  H/8,  W/8)
        x=self.up1_refine(x)       # -> (B, 64, D/8,  H/8,  W/8)

        x=torch.cat([s3,x],dim=1)
        x=self.up2(x)
        x=self.up2_refine(x)

        x=torch.cat([s2,x],dim=1)
        x=self.up3(x)
        x=self.up3_refine(x)

        x=torch.cat([s1,x],dim=1)
        x=self.up4(x)
        x=self.up4_refine(x)

        return x








           
          
