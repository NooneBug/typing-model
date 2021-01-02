import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn.modules.loss import BCEWithLogitsLoss
from typing_model.metrics.my_metrics import MyMetrics
from transformers import BertModel, BertTokenizer
from typing_model.losses.hierarchical_losses import HierarchicalLoss, HierarchicalRegularization

class BaseTyper(pl.LightningModule):
    """
    Base class for typing, defines the metrix and a few things we need to train the networks
    """
    def __init__(self, classes, id2label, label2id, name = "BaseTyper", weights = None, lr=1e-3):
        # TODO: some parameters are not used but are imported in the subclasses
        super().__init__()

        self.id2label = id2label
        self.label2id = label2id
        self.lr = lr

        self.sig = nn.Sigmoid()

        self.weights = weights
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=self.weights)

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

class BaseBERTTyper(BaseTyper):
    def __init__(self, classes, id2label, label2id, name = 'BertTyper', weights = None):
        super().__init__(classes, id2label, label2id, name = 'BertTyper', weights = None)
        self.mention_to_hidden = nn.Linear(1024, 200)
        self.left_to_hidden = nn.Linear(1024, 200)
        self.right_to_hidden = nn.Linear(1024, 200)

        self.hidden_to_output = nn.Linear(600, classes)

    def forward(self, mention, left, right):

        h1 = self.mention_to_hidden(mention)
        h2 = self.left_to_hidden(left)
        h3 = self.right_to_hidden(right)
        concat = torch.cat([h1, h2, h3], dim=1)

        outputs = self.hidden_to_output(concat)

        return outputs

class TransformerBERTTyper(BaseTyper):

    def __init__(self, classes, id2label, label2id, name = 'BertTyper', weights = None):

        super().__init__(classes, id2label, label2id, name, weights)
        # TODO transformer on each embedding (left, right & mention), avgpool on each, separated linear and concat linear
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.mention_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.left_context_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.right_context_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.pooler = nn.AvgPool2d((25, 1))

        self.mention_to_hidden = nn.Linear(768, 200)
        self.left_to_hidden = nn.Linear(768, 200)
        self.right_to_hidden = nn.Linear(768, 200)

        self.hidden_to_output = nn.Linear(600, classes)

        # self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, mention, left, right):

        mention = mention.cuda()
        left = left.cuda()
        right = right.cuda()

        encoded_mention, _ = self.bert(mention)
        te_1 = self.mention_transformer_encoder(encoded_mention)
        pooled_mention = self.pooler(te_1).squeeze()
        h1 = self.mention_to_hidden(pooled_mention)


        encoded_left, _ = self.bert(left)
        te_2 = self.left_context_transformer_encoder(encoded_left)
        pooled_left = self.pooler(te_2).squeeze()
        h2 = self.left_to_hidden(pooled_left)


        encoded_right, _ = self.bert(right)
        te_3 = self.right_context_transformer_encoder(encoded_right)
        pooled_right = self.pooler(te_3).squeeze()
        h3 = self.right_to_hidden(pooled_right)
        concat = torch.cat([h1, h2, h3], dim=1)
        outputs = self.hidden_to_output(concat)

        return outputs

class ConcatenatedContextBERTTyper(BaseTyper):

    def __init__(self, classes, id2label, label2id, max_mention_size=9, max_context_size=40, name = 'BertTyper', weights = None):

        super().__init__(classes, id2label, label2id, name, weights)

        self.context_pooler = nn.AvgPool2d((max_context_size, 1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.context_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.mention_pooler = nn.AvgPool2d((max_mention_size, 1))
        self.mention_to_hidden = nn.Linear(768, 200)
        self.context_to_hidden = nn.Linear(768, 200)
        self.hidden_to_output = nn.Linear(400, classes)

        self.droppy = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, mention, context):

        mention = mention.cuda()
        context = context.cuda()

        encoded_mention = self.bert(mention).last_hidden_state
        pooled_mention = self.mention_pooler(encoded_mention).squeeze()
        h1 = self.droppy(self.relu(self.mention_to_hidden(pooled_mention)))

        encoded_context = self.bert(context).last_hidden_state
        te_2 = self.context_transformer_encoder(encoded_context)
        pooled_context = self.context_pooler(te_2).squeeze()
        h2 = self.droppy(self.relu(self.context_to_hidden(pooled_context)))

        concat = torch.cat([h1, h2], dim=1)
        outputs = self.hidden_to_output(concat)

        return outputs

    def training_step(self, batch, batch_step):
        mention_x, contexts_x, labels = batch

        labels = labels.cuda()

        model_output = self(mention_x, contexts_x)

        loss = self.compute_loss(model_output, labels)

        self.log('train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_step):
        mention_x, contexts_x, labels = batch

        labels = labels.cuda()

        model_output = self(mention_x, contexts_x)

        val_loss = self.compute_loss(model_output, labels)
        
        self.log('val_loss', val_loss, on_epoch=True, on_step=False)

        self.update_metrics(pred=model_output, labels=labels)

        return val_loss

class TransformerWHierarchicalLoss(ConcatenatedContextBERTTyper):

    def __init__(self, classes, id2label, label2id, mode, dependecy_file_path, weights = None, name = 'HierarchicalNet'):
        super().__init__(classes, id2label, label2id, name=name, weights=weights)

        self.classification_loss = BCEWithLogitsLoss(pos_weight=self.weights, reduction='none')
        self.hierarchical_loss = HierarchicalLoss(mode = mode,
                                                    id2label= id2label,
                                                    label2id = label2id,
                                                    label_dependency_file_path = dependecy_file_path)

    def compute_loss(self, pred, true):
        classification_loss = self.classification_loss(pred, true)
        pred = self.sig(pred)
        return  self.hierarchical_loss.compute_loss(classification_loss, pred, true)

class TransformerWHierarchicalRegularization(ConcatenatedContextBERTTyper):
    def __init__(self, classes, id2label, label2id, mode, dependecy_file_path, weights = None, name = 'HierarchicalNet'):
        super().__init__(classes, id2label, label2id, name=name, weights=weights)

        self.classification_loss = BCEWithLogitsLoss(pos_weight=self.weights)
        self.hierarchical_regularitazion = HierarchicalRegularization(mode = mode,
                                                                        id2label= id2label,
                                                                        label2id = label2id,
                                                                        label_dependency_file_path = dependecy_file_path)

    def compute_loss(self, pred, true):
        classification_loss = self.classification_loss(pred, true)
        pred = self.sig(pred)
        regularization_value = self.hierarchical_regularitazion.compute_loss(pred, true)
        self.log('other_metrics/regularization_value', regularization_value)
        return  classification_loss + regularization_value

class OnlyMentionBERTTyper(BaseTyper):

    def __init__(self, classes, id2label, label2id, max_mention_size = 9, name = 'BertTyper', weights = None):

        super().__init__(classes, id2label, label2id, name, weights)

        self.mention_pooler = nn.AvgPool2d((max_mention_size, 1))

        self.mention_to_hidden = nn.Linear(768, 200)

        self.hidden_to_output = nn.Linear(200, classes)

        self.droppy = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, mention):

        mention = mention.cuda()

        encoded_mention = self.bert(mention).last_hidden_state
        pooled_mention = self.mention_pooler(encoded_mention).squeeze()
        h1 = self.droppy(self.relu(self.mention_to_hidden(pooled_mention)))

        outputs = self.hidden_to_output(h1)

        return outputs

    def training_step(self, batch, batch_step):
        mention_x, labels = batch

        labels = labels.cuda()

        model_output = self(mention_x)

        loss = self.compute_loss(model_output, labels)

        self.log('train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_step):
        mention_x, labels = batch

        labels = labels.cuda()

        model_output = self(mention_x)

        val_loss = self.compute_loss(model_output, labels)
        
        # TO DO: log the total val loss, not at each validation step
        self.log('val_loss', val_loss, on_epoch=True, on_step=False)
        
        self.update_metrics(pred=model_output, labels=labels)

        return val_loss

class OnlyContextBERTTyper(BaseTyper):

    def __init__(self, classes, id2label, label2id, max_context_size=40, name = 'BertTyper', weights = None):

        super().__init__(classes, id2label, label2id, name, weights)

        self.context_pooler = nn.AvgPool2d((max_context_size, 1))

        self.context_to_hidden = nn.Linear(768, 200)

        self.hidden_to_output = nn.Linear(200, classes)

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.context_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  

        self.droppy = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, context):

        context = context.cuda()

        encoded_context = self.bert(context).last_hidden_state
        te_2 = self.context_transformer_encoder(encoded_context)
        pooled_context = self.context_pooler(te_2).squeeze()
        h1 = self.droppy(self.relu(self.context_to_hidden(pooled_context)))

        outputs = self.hidden_to_output(h1)

        return outputs

    def training_step(self, batch, batch_step):
        context_x, labels = batch

        labels = labels.cuda()

        model_output = self(context_x)

        loss = self.compute_loss(model_output, labels)

        self.log('train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_step):
        context_x, labels = batch

        labels = labels.cuda()

        model_output = self(context_x)

        val_loss = self.compute_loss(model_output, labels)
        
        # TO DO: log the total val loss, not at each validation step
        self.log('val_loss', val_loss, on_epoch=True, on_step=False)
        
        self.update_metrics(pred=model_output, labels=labels)

        return val_loss
