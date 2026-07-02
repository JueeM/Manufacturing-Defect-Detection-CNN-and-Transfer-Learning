import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_transforms(img_size=128, augment=False):
    #returns (train_transform,eval_transform). resnet expects 3-channel input and normalised states
    # so we convert grayscale to RGB
    imagenet_mean=[0.485,0.456,0.406]
    imagenet_std=[0.229,0.224,0.225]

    eval_tf=transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean,imagenet_std)
    ])

    if augment:
        train_tf=transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2,contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean,imagenet_std)
        ])
    else:
        train_tf=eval_tf
    return train_tf,eval_tf

def load_datasets(data_dir,img_size=128,augment=True):
    train_tf,eval_tf=get_transforms(img_size,augment)
    train_dir=os.path.join(data_dir,"train")
    test_dir=os.path.join(data_dir,"test")
    train_ds=datasets.ImageFolder(train_dir,transform=train_tf)
    test_ds=datasets.ImageFolder(test_dir,transform=eval_tf)

    return train_ds,test_ds

def get_class_weights(dataset):
    targets=np.array(dataset.targets)
    class_counts=np.bincount(targets)
    total=len(targets)
    weights=total/(len(class_counts)*class_counts)
    return torch.tensor(weights,dtype=torch.float32)

def get_dataloaders(train_ds,test_ds, batch_size=32, val_split=0.15, seed=42):
    n_val=int(len(train_ds)*val_split)
    n_train=len(train_ds)-n_val
    g=torch.Generator().manual_seed(seed)
    train_subset, val_subset=torch.utils.data.random_split(train_ds,[n_train,n_val],generator=g)

    train_loader=DataLoader(train_subset,batch_size=batch_size,shuffle=True,num_workers=2)
    val_loader=DataLoader(val_subset,batch_size=batch_size,shuffle=False,num_workers=2)
    test_loader=DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=2)

    return train_loader,val_loader,test_loader

def save_checkpoint(model,path):
    torch.save(model.state_dict(),path)
    print(f"saved model to {path}")
