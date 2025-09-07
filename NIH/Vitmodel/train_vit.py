import os 
import math
import time
import torch
import torch.nn as nn
import pickle

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group
from torch.utils.data import DataLoader,DistributedSampler
from dataset import train_dataset,val_dataset

from contextlib import nullcontext

out_dir="out"
init_from="pretrained" # pretrained or resume or scratch( we will never do this)
per_device_batch_size=4
grad_accumulation_steps=5*2
#-------------------------------wandb logging------------------------------------
wandb_log=False
wandb_project_name="NihChest"
wandb_run_name="vit_scratch"

#---------------------DDP settings---------------------
backend="nccl"
device="cuda"
dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"


# optimizer / schedule
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

compile_model=True



#----------------DDP init-------------------
ddp=int(os.environ.get("RANK",-1)!=-1)
if ddp:
    init_process_group(backend=backend)
    ddp_rank=int(os.environ["RANK"])
    ddp_local_rank=int(os.environ["LOCAL_RANK"])
    ddp_world_size=int(os.environ["WORLD_SIZE"])

    master_process=(ddp_rank==0)
    seed_offset=ddp_local_rank
    device=f"cuda:{ddp_local_rank}"
    
    assert grad_accumulation_steps%ddp_world_size==0
    grad_accumulation_steps//=ddp_world_size

else:
    master_process=True
    seed_offset=0
    ddp_world_size=1

#-----------device/datatype setting --------------

device_type="cuda" if "cuda" in device else "cpu"
ptdtype={"float32":torch.float32,"bfloat16":torch.bfloat16,"float16":torch.float16}[dtype]

ctx=nullcontext() if device_type=="cpu" else torch.autocast(device_type=device_type,enabled=(ptdtype==ptdtype))
scaler=torch.cuda.amp.GradScaler(enabled=(dtype=="float16"))


#--------------------cuda settings---------------
torch.cuda.manual_seed(1472+seed_offset)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudann.allow_tf32=True

#------------get_lr---------
def get_lr(it,learning_rate,min_lr):
    if not decay_lr:
        return learning_rate
    
    if it<warmup_iters:
        return learning_rate*(1+it)/(warmup_iters+1)

    elif  it>lr_decay_iters:
        return min_lr

    else :
        decay_ratio=(it-warmup_iters)/(lr_decay_iters-warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr +coeff* (learning_rate - min_lr)
    



#---------------------get optimizer---------------------
def get_optimizer(model):
    high_lr_params=[]
    low_lr_params=[]

    for p,pn in model.named_parameters():
        if p.startswith("vit.embeddings"):
            high_lr_params.append(pn)

        elif p.startswith('classifier'):
            high_lr_params.append(pn)

        else:
            low_lr_params.append(pn)


    
        # optimizer with two LR groups
    optimizer = torch.optim.AdamW([
        {"params": high_lr_params},  # freshly trained
        {"params": low_lr_params},  # pretrained
    ], weight_decay=weight_decay,betas=(beta1,beta2))
    
    return optimizer

#-------------loading the model-------------------
if init_from=="resume":
    print(f"loading model from {out_dir}")
    ckpt_path=os.path.join(out_dir,"ckpt.pt")

    if os.path.exists(ckpt_path) :
        checkpoint=torch.load(ckpt_path,map_location=device)
    else:
        print("model is not save here")
        
    from loading import model , modelargs    
    
    checkpoint_model_args=checkpoint["model_args"]
    for k,v in modelargs.items():
        modelargs[k]=checkpoint_model_args[k]

    state_dict=checkpoint["model"] 
    model.load_state_dict(state_dict) 
    model=model.to(device)
    # gettting fresh optimizer
    optimizer=get_optimizer(model)
    #getting optimizer state_dict
    optimizer_state_dict=checkpoint["optimizer"]
    # loading optimizer state-dict
    optimizer.load_state_dict(optimizer_state_dict)

         


elif init_from=="pretrained":
    from loading import  model, modelargs
    model=model.to(device)
    optimizer=get_optimizer(model)





#--------------compile--------
if compile_model:
    print("compiling model may take few minutes")
    unoptimized_model=model
    model=torch.compile(model)


#------------------data------------------------
if ddp:
    train_sampler=DistributedSampler(train_dataset,num_replicas=ddp_world_size,shuffle=True)
    val_sampler=DistributedSampler(val_dataset,num_replicas=ddp_world_size)

else:
    train_sampler=None
    val_sampler=None



train_loader=DataLoader(
    train_dataset,
    batch_size=per_device_batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)    

val_loader=DataLoader(
    val_dataset,
    batch_size=per_device_batch_size,
    sampler=val_sampler,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
    
# defineing loss  function --------------------
criterion = nn.BCEWithLogitsLoss()


#--------------------training arguments ---------------------------------------
num_epochs=20
steps_per_epoch=len(train_loader)

max_iters=num_epochs*steps_per_epoch
eval_iters=len(val_loader)

log_interval=max(1,len(train_loader)//100)
save_interval=max(1,len(train_loader)//10)

if wandb_log : 
    import wandb
    wandb.log(project=wandb_project_name,name=wandb_run_name)

@torch.no_grad()
def evaluate_loss(eval_iters):
    model.eval()
    n=0
    total_loss=0.0
    val_iter=iter(val_loader)
    for _ in range(eval_iters):
        batch=next(val_iter,None)
        if batch is None:
            val_iter=iter(val_loader)
            batch=next(val_iter)
        X, Y = batch
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        logits=model(X)
        loss=criterion(logits,Y)
        total_loss+=loss.item()

        n +=1
    avg_loss=total_loss/max(1,n)
    model.train()
    return avg_loss
   

#--------------------training loop-------------------------------
raw_model = model.module if ddp else model # unwrap DDP container if needed
iter_num=0
for epoch in range(num_epochs):
    # important for DistributedSampler shuffling
    if ddp and isinstance(train_sampler,DistributedSampler):
        train_sampler.set_epoch(epoch)

    train_iter=iter(train_loader)
    while True:
        if iter_num>(epoch+1)*steps_per_epoch:
            break  # epoch finished

        optimizer.param_groups[0]["lr"]=get_lr(iter_num,learning_rate=6e-4,min_lr=6e-5)
        optimizer.param_groups[1]["lr"]=get_lr(iter_num,learning_rate=5e-5,min_lr=5e-6)

        t0=time.time()
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(grad_accumulation_steps):
            batch =next(train_iter,None)
            if batch is None:
                #restart the train_loader and shoudln't happen with drop true
                train_iter=iter(train_loader)
                batch= next(train_iter)

            X,Y=batch
            X=X.to(device,non_blocking=True)
            Y=Y.to(device,non_blocking=True)
            # avoid gradient sync on non-final micro steps in DDP
            maybe_no_sync = model.no_sync if (ddp and micro_step < grad_accumulation_steps - 1) else nullcontext
            with maybe_no_sync():
                with ctx:
                    logits = model(X)
                    loss=criterion(logits,Y)
                    loss = loss / grad_accumulation_steps
                    
                scaler.scale(loss).backward()    


        # clip grad,update scaler
        if grad_clip>0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm(model.parameters(),grad_clip)
            scaler.step(optimizer)
            scaler.update()

        t1=time.time()
         # checking for logging 
        if master_process and (iter_num%log_interval==0):
            # quick on-the-fly train loss (last microbatch scaled back)
            train_loss_display = loss.item() * grad_accumulation_steps
            print(f"iter {iter_num} | epoch {epoch+1}/{num_epochs} | loss {train_loss_display:.4f} | {1000*(t1-t0):.0f} ms")
            if wandb_log:
                import wandb
                wandb.log({"train/loss": train_loss_display, "iter": iter_num, "epoch": epoch})
         
        #checking for check pointing
        if master_process and (iter_num%save_interval==0):
            val_loss = evaluate_loss(eval_iters=eval_iters)
            print(f"[eval] epoch {epoch+1}: val/loss {val_loss:.4f}")
            if wandb_log:
                wandb.log({"val/loss": val_loss, "epoch": epoch})



             # save checkpoint once per epoch
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": modelargs,
                "iter_num": iter_num,
                "epoch": epoch,
            }
            torch.save(ckpt, os.path.join(out_dir, f"ckpt_epoch{epoch+1}.pt"))
            # also keep a rolling "latest"
            torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

        iter_num += 1

            
# ----------------- Cleanup --------------------------------
if ddp:
    destroy_process_group()





    
    




































