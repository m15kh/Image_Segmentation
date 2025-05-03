import os
import sys
import time
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
from tqdm import tqdm
import yaml
from torch.utils.data import DataLoader

from scripts.models.net import BuildUnet
from scripts.models.loss import DiceBCELoss
from scripts.models.dataloader import Segmentation_Dataset
from scripts.models.utils import epoch_time
from scripts.models.modules import TrainBaseModule


class SegmentationTrain(TrainBaseModule):
    def __init__(self,
                  image_path:str, 
                  mask_path:str,
                  image_extension:str, 
                  mask_extension:str, 
                  height:int, 
                  width:int,
                  batch_size:int,
                  epochs:int,
                  lr:float, 
                  checkpoint_path:str, 
                  experiment_path:str, 
                  val_percent=0.2,
                  pretrain_path: str = None):  # Add pretrain_path parameter
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = float(lr)
        self.checkpoint_path = checkpoint_path
        self.experiment_path = experiment_path
        self.val_percent = val_percent
        self.pretrain_path = pretrain_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_parameters()
        self.train_loader, self.valid_loader = self._train_val_test_dataloader()

    def _set_parameters(self):
        # self.checkpoint_path = os.path.join(ROOT_DIR, self.checkpoint_path)
        # self.experiment_path = os.path.join(ROOT_DIR, self.experiment_path)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.experiment_path, exist_ok=True)
        # image_path = os.path.join(ROOT_DIR, self.image_path)
        # mask_path = os.path.join(ROOT_DIR, self.mask_path)
        assert os.path.isdir(self.image_path), "train set should be directory"
        assert os.path.isdir(self.mask_path), "train set should be directory"
        self.image_list = []
        self.mask_list = []
        for p in Path(self.image_path).glob("**/*" + self.image_extension):
            self.image_list.append(p)
            mask_file = Path(os.path.join(self.mask_path, p.stem + "." + self.mask_extension))
            self.mask_list.append(mask_file)
            assert os.path.isfile(self.image_list[-1].as_posix()), f"can not find {self.image_list[-1].as_posix()}"
            assert os.path.isfile(self.mask_list[-1].as_posix()), f"can not find corresponding mask file of the image {self.image_list[-1].as_posix()}"
        self.model = BuildUnet().to(self.device)
        
        # Load pre-trained weights if provided
        if self.pretrain_path and os.path.isfile(self.pretrain_path):
            print(f"Loading pre-trained weights from {self.pretrain_path}")
            checkpoint = torch.load(self.pretrain_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.loss_fn = DiceBCELoss()
        self.best_valid_loss = float("inf")
        self.train_losses = []
        self.valid_losses = []

    def _train_val_test_dataloader(self):
        val_len = int(len(self.image_list) * self.val_percent)
        train_len = len(self.image_list) - val_len

        train_x = self.image_list[:train_len]
        train_y = self.mask_list[:train_len]
        valid_x = self.image_list[train_len:]
        valid_y = self.mask_list[train_len:]

        train_dataset = Segmentation_Dataset(train_x, train_y, width=self.width, height=self.height)
        valid_dataset = Segmentation_Dataset(valid_x, valid_y, width=self.width, height=self.height)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        return train_loader, valid_loader

    def _train_one_step(self, x:torch.tensor, y:torch.tensor) -> float:
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _calculate_accuracy(self, y_pred, y):
        """Calculate accuracy by comparing predictions with ground truth."""
        y_pred = (y_pred > 0.5).float()  # Convert probabilities to binary predictions
        correct = (y_pred == y).float().sum()
        total = y.numel()
        return correct / total

    def _train_one_epoch(self, epoch: int) -> tuple:
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}/{self.epochs}', unit='batch') as t:
            for x, y in self.train_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                loss = self._train_one_step(x, y)
                epoch_train_loss += loss

                # Calculate accuracy
                y_pred = self.model(x)
                accuracy = self._calculate_accuracy(y_pred, y)
                epoch_train_accuracy += accuracy.item()

                t.set_postfix(train_loss=epoch_train_loss / len(t), train_acc=epoch_train_accuracy / len(t))
                t.update(1)
        return epoch_train_loss / len(self.train_loader), epoch_train_accuracy / len(self.train_loader)

    def train(self):
        print(f"Dataset Size:\nTrain: {len(self.train_loader.dataset)} - Valid: {len(self.valid_loader.dataset)}\n")

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss, train_accuracy = self._train_one_epoch(epoch)
            valid_loss, valid_accuracy = self.validation_evaluation()
            
            if valid_loss < self.best_valid_loss:
                checkpoint_file_name = os.path.join(self.checkpoint_path, f"segmentation_epoch_{epoch}_loss_{round(valid_loss, 3)}.pth")
                data_str = f"Valid loss improved from {self.best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {self.checkpoint_path}"
                print(data_str)
                self.best_valid_loss = valid_loss
                # Save generator state with additional parameters
                checkpoint_gen = {  
                    'height': self.height,  
                    'width': self.width,
                    'state_dict': self.model.state_dict()
                }

                torch.save(checkpoint_gen, checkpoint_file_name)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            data_str = f'Epoch: {epoch} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
            data_str += f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy:.3f}\n'
            data_str += f'\tValidation Loss: {valid_loss:.3f} | Validation Accuracy: {valid_accuracy:.3f}\n'
            print(data_str)

            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.scheduler.step(valid_loss)

    def validation_evaluation(self):
        self.model.eval()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        with torch.no_grad():
            for x, y in self.valid_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                epoch_loss += loss.item()

                # Calculate accuracy
                accuracy = self._calculate_accuracy(y_pred, y)
                epoch_accuracy += accuracy.item()
        return epoch_loss / len(self.valid_loader), epoch_accuracy / len(self.valid_loader)

    def test_evaluation(self, test_loader):
        return self.validation_evaluation(test_loader)



if __name__ == "__main__":
    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    train_params = yaml.safe_load(open(os.path.join(ROOT_DIR, "params/params_train.yaml")))["segmentation_fingerprint"]
    
    
    train_segmentation = SegmentationTrain(
        image_path=train_params["image_path"],
        mask_path=train_params["mask_path"],
        image_extension=train_params["image_extention"],
        mask_extension=train_params["mask_extention"],
        height=train_params["height"],
        width=train_params["width"],
        batch_size=train_params["batch_size"],
        epochs=train_params["epochs"],
        lr=train_params["learning_rate"],
        checkpoint_path=train_params["checkpoint_path"],
        experiment_path=train_params["experiment_path"],
        val_percent=train_params.get("validation_ratio", 0.2),  # Use the validation ratio from YAML
        pretrain_path=train_params.get("pretrain_path", None))
    train_segmentation.train()
