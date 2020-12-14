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

        # Declare Metrics
        self.micro_precision = pl.metrics.classification.precision_recall.Precision(average='micro', multilabel=True)
        self.micro_recall = pl.metrics.classification.precision_recall.Recall(average='micro', multilabel=True)
        self.micro_f1 = pl.metrics.classification.F1(average='micro', multilabel=True)

        self.macro_precision = pl.metrics.classification.precision_recall.Precision(average='macro', multilabel=True)
        self.macro_recall = pl.metrics.classification.precision_recall.Recall(average='macro', multilabel=True)
        self.macro_f1 = pl.metrics.classification.F1(average='macro', multilabel=True)
        

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
        
        # TO DO: log the total val loss, not at each validation step
        self.log('val_loss', val_loss)
        
        self.update_metrics(pred=model_output, labels=labels)

        return val_loss

    def validation_epoch_end(self, out):
        self.compute_metrics()

    def forward(self, mention, left, right):

        h1 = self.mention_to_hidden(mention)
        h2 = self.left_to_hidden(left)
        h3 = self.right_to_hidden(right)
        concat = torch.cat([h1, h2, h3], dim=1)

        outputs = self.hidden_to_output(concat)

        return outputs

    def compute_loss(self, pred, true):
        return self.classification_loss(pred, true)

    def update_metrics(self, pred, labels):
        self.micro_f1.update(preds=pred, target=labels)
        self.micro_precision.update(preds=pred, target=labels)
        self.micro_recall.update(preds=pred, target=labels)

        self.macro_f1.update(preds=pred, target=labels)
        self.macro_precision.update(preds=pred, target=labels)
        self.macro_recall.update(preds=pred, target=labels)

        #TO DO: compute average prediction number, average void predictions

    def compute_metrics(self):
        self.log('micro_f1', self.micro_f1.compute())
        self.log('micro_p', self.micro_precision.compute())
        self.log('micro_r', self.micro_recall.compute())

        self.log('macro_f1', self.macro_f1.compute())
        self.log('macro_p', self.macro_precision.compute())
        self.log('macro_r', self.macro_recall.compute())