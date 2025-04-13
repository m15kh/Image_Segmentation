import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_data
import torch

from net import AutoEncoder  # Assuming the model is defined in model.py

# Load the data
_, val_dl = load_data()
print(val_dl)
# Load the model
model = AutoEncoder(40).to('cuda')
model.load_state_dict(torch.load('/home/ubuntu/m15kh/own/book/Gans/autoencoder/vanilla_autoencoder/checkpoints/model-simple-autoencoder.pth'))
model.eval()

# Assuming val_ds is the dataset used in the DataLoader
val_ds = val_dl.dataset
print(val_ds)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for i in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    im = im.to(device)
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(im[0].cpu().detach().numpy(), cmap='gray')
    ax[0].set_title('input')
    ax[0].axis('off')
    ax[1].imshow(_im[0].cpu().detach().numpy(), cmap='gray')
    ax[1].set_title('prediction')
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'img-output/output_{i}.png')
    plt.close()

print("finish")