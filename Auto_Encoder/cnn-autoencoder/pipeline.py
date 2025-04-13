import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

#local 
from net import ConvAutoEncoder
from dataloader import load_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'






class ModelTrainer:
    def __init__(self, model, criterion, optimizer, log_dir='/home/ubuntu/m15kh/own/book/Gans/autoencoder/cnn-autoencoder/checkpoints/runs'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = SummaryWriter(log_dir)


    def train_batch(self, input, step):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.criterion(output, input)
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('Train/Loss', loss.item(), step)

        return loss

    @torch.no_grad()
    def validate_batch(self, input, step):
        self.model.eval()
        output = self.model(input)
        loss = self.criterion(output, input)
        self.writer.add_scalar('Validation/Loss', loss.item(), step)
        return loss

    def train(self, trn_dl, val_dl, num_epochs, log):
        step = 0

        for epoch in range(num_epochs):
            N = len(trn_dl)
            for ix, (data, _) in enumerate(trn_dl):
                loss = self.train_batch(data,step)
                step += 1
                log.record(pos=(epoch + (ix+1)/N), trn_loss=loss, end='\r')

            N = len(val_dl)
            for ix, (data, _) in enumerate(val_dl):
                loss = self.validate_batch(data,step)
                step += 1
                log.record(pos=(epoch + (ix+1)/N), val_loss=loss, end='\r')
            log.report_avgs(epoch+1)
        
        log.plot_epochs(log=True)
        self.writer.close()

        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    # def load_model(self, path):
    #     self.model.load_state_dict(torch.load(path))
    #     self.model.eval()
    #     print(f"Model loaded from {path}")

class Report:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.trn_losses = []
        self.val_losses = []

    def record(self, pos, trn_loss=None, val_loss=None, end='\n'):
        if trn_loss is not None:
            self.trn_losses.append((pos, trn_loss.item()))
        if val_loss is not None:
            self.val_losses.append((pos, val_loss.item()))
        print(f'Epoch [{int(pos)}], Step [{pos - int(pos):.4f}], '
              f'Train Loss: {trn_loss.item() if trn_loss else "N/A"}, '
              f'Val Loss: {val_loss.item() if val_loss else "N/A"}', end=end)

    def report_avgs(self, epoch):
        avg_trn_loss = sum([x[1] for x in self.trn_losses if int(x[0]) == epoch]) / len(
            [x[1] for x in self.trn_losses if int(x[0]) == epoch])
        avg_val_loss = sum([x[1] for x in self.val_losses if int(x[0]) == epoch]) / len(
            [x[1] for x in self.val_losses if int(x[0]) == epoch])
        print(f'Epoch [{epoch}], Avg Train Loss: {avg_trn_loss}, Avg Val Loss: {avg_val_loss}')

    def plot_epochs(self, log=True, filename='img-output/loss_plot.png'):
        trn_x, trn_y = zip(*self.trn_losses)
        val_x, val_y = zip(*self.val_losses)
        plt.plot(trn_x, trn_y, label='Train Loss')
        plt.plot(val_x, val_y, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(filename)
        plt.close()

trn_dl, val_dl = load_data()
model = ConvAutoEncoder().to(device)
criterion = nn.MSELoss()    
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
num_epoches = 6
log = Report(num_epoches)

trainer = ModelTrainer(model, criterion, optimizer)
trainer.train(trn_dl, val_dl, num_epoches, log)
trainer.save_model('/home/ubuntu/m15kh/own/book/Gans/autoencoder/cnn-autoencoder/checkpoints/model-convautoencoder.pth')