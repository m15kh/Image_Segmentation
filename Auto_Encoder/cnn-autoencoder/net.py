import torch.nn as nn
import torch
from torchsummary import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=1),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# model = ConvAutoEncoder().to(device)
# summary(model, torch.zeros(2,1,28,28))
 
