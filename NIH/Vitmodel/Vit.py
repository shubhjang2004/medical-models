

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

"""
ViTForImageClassification(
  (vit): ViTModel(
    (embeddings): ViTEmbeddings(
      (patch_embeddings): ViTPatchEmbeddings(
        (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      )
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): ViTEncoder(
      (layer): ModuleList(
        (0-11): 12 x ViTLayer(
          (attention): ViTAttention(
            (attention): ViTSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
            )
            (output): ViTSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): ViTIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ViTOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  )
  (classifier): Linear(in_features=768, out_features=1000, bias=True)
)
"""
#----------------------making Vit from scratch--------------------------------------
@dataclass
class Modelargs:
    dim:int=768
    dropout:int=0.0
    n_heads:int=12
    patch_size:int=32
    
    n_layers:int=12
    in_channels: int = 3
    img_size: int = 1024
    num_patches: int = (img_size // patch_size) ** 2
    num_classes:int=15



class ViTForImageClassification(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.vit=ViTModel(args)
        self.classifier=nn.Linear(args.dim,args.num_classes)

        self.apply(self.weight_init)

    def forward(self,x):
        x=self.vit(x)
        # x: (batch, num_patches+1, dim)        
        cls_token = x[:, 0]                   # take [CLS] embedding       
        logits=self.classifier(cls_token)           # logits: (batch, num_classes)
        
        return logits


        #----------weight initialization-------------------------
    def weight_init(self,Module):
        if isinstance(Module,nn.Linear):
            torch.nn.init.trunc_normal_(Module.weight,mean=0.0,std=0.02)
            if Module.bias is not None:
                torch.nn.init.zeros_(Module.bias)
        elif isinstance(Module,nn.Embedding):
            torch.nn.init.trunc_normal_(Module.weight,mean=0.0,std=0.02)
        elif isinstance(Module,nn.Conv2d):
            torch.nn.init.trunc_normal_(Module.weight,mean=0.0,std=0.02)
            if Module.bias is not None:
                torch.nn.init.zeros_(Module.bias)



class ViTModel(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        
        self.embeddings=ViTEmbeddings(args)
        self.encoder=ViTEncoder(args)
        self.layernorm=nn.LayerNorm(args.dim)
        
        
       

    def forward(self,x):
        x=self.embeddings(x)
        x=self.encoder(x)
        x=self.layernorm(x)
       
        return x
    




class ViTEncoder(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.layer=nn.ModuleList([ViTLayer(args) for _ in range(args.n_layers)])

    def forward(self,x):
        for layer in self.layer:
            x=layer(x)
        return  x 



class ViTLayer(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.attention=ViTAttention(args)
        self.intermediate=ViTIntermediate(args)
        self.output=ViTOutput(args)
        self.layernorm_before=nn.LayerNorm(args.dim)
        self.layernorm_after=nn.LayerNorm(args.dim)

    def forward(self,x):
        x=x+self.attention(self.layernorm_before(x))
        x=x+self.output(self.intermediate(self.layernorm_after(x)))

        return x 



class ViTAttention(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.attention=ViTSelfAttention(args)
        self.output=ViTSelfOutput(args)
    def forward(self,x):
        x=self.attention(x)
        x=self.output(x)
        return x 



class  ViTSelfAttention(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.query=nn.Linear(args.dim,args.dim)
        self.key=nn.Linear(args.dim,args.dim)
        self.value=nn.Linear(args.dim,args.dim)
        self.n_heads=args.n_heads
        self.head_dim=args.dim//args.n_heads     

    
        self.softmax_scale = self.head_dim**-0.5


    def forward(self,x):
        B,T,C=x.size()

        assert C%self.n_heads==0, f"dimension mismatch"

        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        query=q.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        key=k.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        value=v.view(B,T,self.n_heads,self.head_dim).transpose(1,2)

        attn=query@key.transpose(-2,-1)*self.softmax_scale

        
        attn = F.softmax(attn, dim=-1)   # <--- missing
        output=attn@value
        output=output.transpose(1,2).contiguous().view(B,T,C)
        return output

class ViTSelfOutput(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.dense=nn.Linear(args.dim,args.dim)
        self.dropout=nn.Dropout(args.dropout)

    def forward(self,x):
        x=self.dropout(self.dense(x))
        return x

class ViTIntermediate(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.dense=nn.Linear(args.dim,4*args.dim)
        self.intermediate_act_fn=nn.GELU()

    def forward(self,x):
        x=self.intermediate_act_fn(self.dense(x))
        return x

class ViTOutput(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.dense=nn.Linear(4*args.dim,args.dim)
        self.dropout=nn.Dropout(args.dropout)

    def forward(self,x):
        return self.dropout(self.dense(x))
                           

class ViTEmbeddings(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.patch_embeddings=ViTPatchEmbeddings(args)
                # 2. Learnable [CLS] token (special token for classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.dim))

        # 3. Learnable positional embeddings (for CLS + all patches)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, args.num_patches + 1, args.dim)
        )
        self.dropout=nn.Dropout(args.dropout)

    def forward(self,x):
        # Step A. Project image into patch embeddings
        embeddings = self.patch_embeddings(x)  
        # shape: (B, num_patches, hidden_size)
        # Step B. Expand CLS token to batch size
        cls_tokens = self.cls_token.expand(embeddings.size(0), -1, -1)  
        # shape: (B, 1, hidden_size)
        # Step C. Concatenate CLS token in front of patch embeddings
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  
        # shape: (B, num_patches+1, hidden_size)
        # Step D. Add positional embeddings (elementwise addition)
        embeddings = embeddings + self.position_embeddings  
        # shape: (B, num_patches+1, hidden_size)
        # Step E. Apply dropout (regularization)
        embeddings = self.dropout(embeddings)
        # Final output → goes into the Transformer encoder
        return embeddings
    


    

class ViTPatchEmbeddings(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.projection=nn.Conv2d(args.in_channels,args.dim,kernel_size=args.patch_size,stride=args.patch_size)

    def forward(self,x):
        # x: (batch, 3, H, W)
        x=self.projection(x) # (batch, embed_dim, H/P, W/P)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x
    

"""
class Classifier(nn.Module):
    def __init__(self,dim:int,num_classes:int):
        super().__init__()
        self.layernorm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim,num_classes)

    def forward(self,x):
        # x: (batch, num_patches+1, dim)
        
        cls_token = x[:, 0]                   # take [CLS] embedding
        cls_token = self.layernorm(cls_token)
        return self.head(cls_token)           # logits: (batch, num_classes)


class Vit_final_model(nn.Module):
    def __init__(self,model:nn.Module,classifier:nn.Module):
        super().__init__()
        self.model=model
        self.classifier=classifier


    def forward(self,x):
        x=self.model(x)
        logits=self.classifier(x)
        return logits    

class ViTPooler(nn.Module):
    def __init__(self,args:Modelargs):
        super().__init__()
        self.dense=nn.Linear(args.dim,args.dim)
        self.activation=nn.Tanh()    
    
    def forward(self,x):
        return self.activation(self.dense(x))

"""        


