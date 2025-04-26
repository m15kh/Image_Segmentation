from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

def train_unet(model, train_loader, val_loader, epochs, lr=1e-4, device='cuda'):
    criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")

        # Validation - only run if val_loader is provided
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                print(f"Validation Loss: {val_loss / len(val_loader)}")
