import torch
from torch import nn
import pytorch_lightning as pl
from typing_model.metrics.my_metrics import MyMetrics

class BaseBERTTyper(pl.LightningModule):
    def __init__(self, classes, id2label, label2id, name = 'BertTyper', weights = None):
        super().__init__()

        self.id2label = id2label
        self.label2id = label2id

        self.mention_to_hidden = nn.Linear(1024, 200)
        self.left_to_hidden = nn.Linear(1024, 200)
        self.right_to_hidden = nn.Linear(1024, 200)

        self.hidden_to_output = nn.Linear(600, classes)

        self.sig = nn.Sigmoid()

        self.weights = weights
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=self.weights)


        # Declare Metrics
        self.micro_precision = pl.metrics.classification.precision_recall.Precision(num_classes=len(self.id2label),
                                                                                    average='micro', 
                                                                                    multilabel=True)
        self.micro_recall = pl.metrics.classification.precision_recall.Recall(num_classes=len(self.id2label),
                                                                                average='micro', 
                                                                                multilabel=True)
        self.micro_f1 = pl.metrics.classification.F1(num_classes=len(self.id2label),
                                                        average='micro', 
                                                        multilabel=True)

        self.macro_precision = pl.metrics.classification.precision_recall.Precision(num_classes=len(self.id2label),
                                                                                    average='macro', 
                                                                                    multilabel=True)
        
        self.macro_recall = pl.metrics.classification.precision_recall.Recall(num_classes=len(self.id2label),
                                                                                    average='macro', 
                                                                                    multilabel=True)

        self.macro_f1 = pl.metrics.classification.F1(num_classes=len(self.id2label),
                                                        average='macro', 
                                                        multilabel=True)

        self.my_metrics = MyMetrics(id2label=id2label)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_step):
        mention_x, left_x, right_x, labels = batch

        model_output = self(mention_x, left_x, right_x)

        loss = self.compute_loss(model_output, labels)

        self.log('train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_step):
        mention_x, left_x, right_x, labels = batch

        model_output = self(mention_x, left_x, right_x)

        val_loss = self.compute_loss(model_output, labels)
        
        # TO DO: log the total val loss, not at each validation step
        self.log('val_loss', val_loss, on_epoch=True, on_step=False)
        
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
        
        pred = self.sig(pred)

        self.micro_f1.update(preds=pred, target=labels)
        self.micro_precision.update(preds=pred, target=labels)
        self.micro_recall.update(preds=pred, target=labels)

        self.macro_f1.update(preds=pred, target=labels)
        self.macro_precision.update(preds=pred, target=labels)
        self.macro_recall.update(preds=pred, target=labels)

        self.my_metrics.update(preds=pred, target=labels)

        #TO DO: compute average prediction number, average void predictions

    def compute_metrics(self):
        self.log('micro/micro_f1', self.micro_f1.compute())
        self.log('micro/micro_p', self.micro_precision.compute())
        self.log('micro/micro_r', self.micro_recall.compute())

        self.log('macro/macro_f1', self.macro_f1.compute())
        self.log('macro/macro_p', self.macro_precision.compute())
        self.log('macro/macro_r', self.macro_recall.compute())

        avg_pred_number, void_predictions, _, _, _, ma_p, ma_r, ma_f1 = self.my_metrics.compute()

        self.log('example_macro/macro_f1', ma_f1)
        self.log('example_macro/macro_p', ma_p)
        self.log('example_macro/macro_r', ma_r)

        self.log('other_metrics/avg_pred_number', avg_pred_number)
        self.log('other_metrics/void_predictions', void_predictions)
