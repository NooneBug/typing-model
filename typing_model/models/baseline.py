import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class BaseBERTTyper(pl.LightningModule):
    def __init__(self, classes, id2label, label2id, name = 'BertTyper'):
        super().__init__()

        self.id2label = id2label
        self.label2id = label2id
        self.classification_loss = nn.BCEWithLogitsLoss()

        self.mention_to_hidden = nn.Linear(1024, 200)
        self.left_to_hidden = nn.Linear(1024, 200)
        self.right_to_hidden = nn.Linear(1024, 200)

        self.hidden_to_output = nn.Linear(600, classes)

        self.sig = nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_step):
        mention_x, left_x, right_x, labels = batch

        model_output = self(mention_x, left_x, right_x)

        loss = self.compute_loss(model_output, labels)

        return loss

    def validation_step(self, batch, batch_step):
        mention_x, left_x, right_x, labels = batch

        model_output = self(mention_x, left_x, right_x)

        val_loss = self.compute_loss(model_output, labels)
        self.log('val_loss', val_loss)

        return val_loss

    def forward(self, mention, left, right):

        h1 = self.mention_to_hidden(mention)
        h2 = self.left_to_hidden(left)
        h3 = self.right_to_hidden(right)
        concat = torch.cat([h1, h2, h3], dim=1)

        outputs = self.hidden_to_output(concat)

        return outputs

    def compute_loss(self, pred, true):
        return self.classification_loss(pred, true)
