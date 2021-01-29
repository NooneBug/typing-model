from typing_model.metrics.my_metrics import MyMetrics
import pytorch_lightning as pl
from torch import nn
import torch

class BaseTyper(pl.LightningModule):
    """
    Base class for typing, defines the metrix and a few things we need to train the networks
    """
    def __init__(self, classes, id2label, label2id, name = "BaseTyper", weights = None, lr=1e-3, loss_multiplier = 1):
        # TODO: some parameters are not used but are imported in the subclasses
        super().__init__()

        self.id2label = id2label
        self.label2id = label2id
        self.lr = lr

        self.sig = nn.Sigmoid()

        self.weights = weights
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=self.weights)

        self.loss_multiplier = loss_multiplier

        # Declaring evaluation Metrics
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_step):
        mention_x, left_x, right_x, labels = batch
        labels = labels.cuda()
        model_output = self(mention_x, left_x, right_x)
        loss = self.compute_loss(model_output, labels)
        self.log('train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_step):
        mention_x, left_x, right_x, labels = batch
        labels = labels.cuda()
        model_output = self(mention_x, left_x, right_x)
        val_loss = self.compute_loss(model_output, labels)
        # TODO: log the total val loss, not at each validation step
        self.log('val_loss', val_loss, on_epoch=True, on_step=False)
        self.update_metrics(pred=model_output, labels=labels)

        return val_loss

    def validation_epoch_end(self, out):
        self.compute_metrics()

    def forward(self, *args, **kwargs):
        raise Exception("Declare a forward which takes in input the mention representation and its contexts")

    def compute_loss(self, pred, true):
        return self.classification_loss(pred, true) * self.loss_multiplier
    
    def get_discrete_pred(self, pred, threshold = 0.5):
        mask = pred > threshold

        ones = torch.ones(mask.shape).cuda()
        zeros = torch.zeros(mask.shape).cuda()

        discrete_pred = torch.where(mask, ones, zeros)

        max_values_and_indices = torch.max(pred, dim = 1)

        for dp, i in zip(discrete_pred, max_values_and_indices.indices):
            dp[i] = 1
        
        return discrete_pred

    def update_metrics(self, pred, labels):

        pred = self.sig(pred)

        pred = self.get_discrete_pred(pred)

        self.micro_f1.update(preds=pred, target=labels)
        self.micro_precision.update(preds=pred, target=labels)
        self.micro_recall.update(preds=pred, target=labels)

        self.macro_f1.update(preds=pred, target=labels)
        self.macro_precision.update(preds=pred, target=labels)
        self.macro_recall.update(preds=pred, target=labels)

        self.my_metrics.update(preds=pred, target=labels)


    def compute_metrics(self):
        self.log('micro/micro_f1', self.micro_f1.compute())
        self.log('micro/micro_p', self.micro_precision.compute())
        self.log('micro/micro_r', self.micro_recall.compute())

        self.log('macro/macro_f1', self.macro_f1.compute())
        self.log('macro/macro_p', self.macro_precision.compute())
        self.log('macro/macro_r', self.macro_recall.compute())

        avg_pred_number, void_predictions, _, _, _, ma_p, ma_r, ma_f1, predicted_class_number = self.my_metrics.compute()

        self.log('example_macro/macro_f1', ma_f1)
        self.log('example_macro/macro_p', ma_p)
        self.log('example_macro/macro_r', ma_r)

        self.log('other_metrics/avg_pred_number', avg_pred_number)
        self.log('other_metrics/void_predictions', void_predictions)
        self.log('other_metrics/predicted_class_number', predicted_class_number)