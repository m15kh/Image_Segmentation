import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, dim_latent):
        super(AutoEncoder, self).__init__()
        self.dim_latent = dim_latent
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # The input should be flattened to (batch_size, 28*28)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, dim_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input to (batch_size, 28*28)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 28, 28)  # Reshape output to (batch_size, 1, 28, 28)
        return x

# model = AutoEncoder(3).to(device)
# summary(model, torch.zeros(3, 28, 28))
