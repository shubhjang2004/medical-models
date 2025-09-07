from Vit import ViTForImageClassification,Modelargs


modelargs=Modelargs()
model=ViTForImageClassification(modelargs)



def load_pretrained_vit(model):
    #---------loading pretrained vit----------------
    from transformers import ViTForImageClassification
    model_name="google/vit-base-patch16-224"
    model_hf= ViTForImageClassification.from_pretrained(model_name)

    #------------state_dict-----------
    sd_hf=model_hf.state_dict()
    sd=model.state_dict()
    sd_hf_keys=sd_hf.keys()
    sd_keys=sd.keys()
    if ((list(sd_keys)==list(sd_hf_keys))):
        print("all keys matched ")
    else:
        print("keys didn't match")    
    
        
    load_sd_hf_keys=[k for k in sd_hf_keys if not k.startswith("vit.embeddings")]
    load_sd_hf_keys=[k for k in load_sd_hf_keys if not k.startswith("classifier")]

    for k in load_sd_hf_keys:
        sd[k]=sd_hf[k]

    model.load_state_dict(sd)

    return model

        

model=load_pretrained_vit(model)



