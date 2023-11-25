import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
def root_mean_squared_log_error(y_true, y_pred):
    log1p_true = torch.log1p(y_true)
    log1p_pred = torch.log1p(y_pred)
    
    squared_log_errors = (log1p_true - log1p_pred) ** 2
    mean_squared_log_error = torch.mean(squared_log_errors)
    
    return torch.sqrt(mean_squared_log_error)

class FNN(pl.LightningModule):
    def __init__(self, input_size):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(y_hat)
        loss = root_mean_squared_log_error(y, y_hat)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = root_mean_squared_log_error(y, y_hat)
        self.validation_step_outputs.append(loss)
        return loss
    def test_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = root_mean_squared_log_error(y, y_hat)
        return loss
    
    def on_train_epoch_end(self):
        loss = torch.stack(self.training_step_outputs).mean()
        print(f"{loss.item():>4f}")
        self.training_step_outputs.clear
    
    def on_valid_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        print(f"[VALID LOSS]: {loss.item():>4f}")
        self.validation_step_outputs.clear

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
