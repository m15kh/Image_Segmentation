from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data():
    img_transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5]),
     transforms.Lambda(lambda x: x.to(device))
     ])


    trn_ds = MNIST('/home/ubuntu/m15kh/own/book/Gans/autoencoder/content/', transform=img_transform, train=True, download=True)
    val_ds = MNIST('/home/ubuntu/m15kh/own/book/Gans/autoencoder/content/', transform=img_transform, train=False, download=True)
    
    batch_size = 128
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return trn_dl, val_dl
