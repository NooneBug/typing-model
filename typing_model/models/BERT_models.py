import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.loss import BCEWithLogitsLoss
from transformers import BertModel
from typing_model.data.BERT_datasets import ConcatenatedContextTypingBERTDataSet
from typing_model.losses.hierarchical_losses import HierarchicalLoss, HierarchicalRegularization
from typing_model.models.base_models import BaseTyper

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

class BertEncoder(nn.Module):

    def __init__(self, bert_fine_tuning):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if bert_fine_tuning:
            for name, param in self.bert.named_parameters():
                if '11' not in name:
                    param.requires_grad = False
        else:
            for _, param in self.bert.named_parameters():
                param.requires_grad = False
    
    def forward(self, input, mention_or_context):
        b_encoding = self.bert(input).last_hidden_state

        return b_encoding


class ConcatenatedContextBERTTyper(BaseTyper):

    def __init__(self, classes, id2label, label2id, max_mention_size=9, max_context_size=16, name = 'BertTyper', 
                    weights = None, lr = None, bert_fine_tuning = None, loss_multiplier = 1):

        super().__init__(classes, id2label, label2id, name, weights, lr, loss_multiplier)

        self.input_encoder = BertEncoder(bert_fine_tuning)
        self.context_pooler = nn.AvgPool2d((max_context_size, 1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.context_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.context_to_hidden = nn.Linear(768, 200)

        self.mention_pooler = nn.AvgPool2d((max_mention_size, 1))
        self.mention_to_hidden = nn.Linear(768, 200)

        self.hidden_to_hidden = Linear(400, 400).cuda()
        self.hidden_to_output = nn.Linear(400, classes)

        self.droppy = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def fine_tuning_setup(self, new_class_number):
        for param in self.input_encoder.bert.parameters():
            param.requires_grad = False

        print('substituting the final layer with a layer with {} classes'.format(new_class_number))
        self.hidden_to_output = nn.Linear(400, new_class_number)


    def forward(self, mention, context):
        mention = mention.cuda()
        context = context.cuda()

        encoded_mention = self.input_encoder(mention, 'mention')
        pooled_mention = self.mention_pooler(encoded_mention).squeeze()
        h1 = self.droppy(self.relu(self.mention_to_hidden(pooled_mention)))

        encoded_context = self.input_encoder(context, 'context')
        te_2 = self.context_transformer_encoder(encoded_context)
        pooled_context = self.context_pooler(te_2).squeeze()
        h2 = self.droppy(self.relu(self.context_to_hidden(pooled_context)))

        concat = torch.cat([h1, h2], dim=1)
        h = self.droppy(self.relu(self.hidden_to_hidden(concat)))
        outputs = self.hidden_to_output(h)

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

class Bertv2(ConcatenatedContextBERTTyper):
    def __init__(self, classes, id2label, label2id, max_mention_size=9, max_context_size=16, name = 'BertTyper', 
                    weights = None, lr = None, bert_fine_tuning = None, loss_multiplier = 1):
        
        super().__init__(classes=classes, id2label=id2label, label2id=label2id, max_mention_size=max_mention_size, 
                            max_context_size =max_context_size, name = name,  weights= weights, lr = lr, 
                            bert_fine_tuning = bert_fine_tuning, loss_multiplier = loss_multiplier)

        self.mention_batch_n = nn.BatchNorm1d(200)
        self.context_batch_n = nn.BatchNorm1d(200)
        self.cat_batch_n = nn.BatchNorm1d(400)
        self.hidden_batch_n = nn.BatchNorm1d(400)

    def forward(self, mention, context):
        mention = mention.cuda()
        context = context.cuda()

        encoded_mention = self.input_encoder(mention, 'mention')
        pooled_mention = self.mention_pooler(encoded_mention).squeeze()
        h1 = self.droppy(self.mention_batch_n(self.relu(self.mention_to_hidden(pooled_mention))))

        encoded_context = self.input_encoder(context, 'context')
        te_2 = self.context_transformer_encoder(encoded_context)
        pooled_context = self.context_pooler(te_2).squeeze()
        h2 = self.droppy(self.context_batch_n(self.relu(self.context_to_hidden(pooled_context))))

        concat = torch.cat([h1, h2], dim=1)
        h = self.droppy(self.hidden_batch_n(self.relu(self.hidden_to_hidden(concat))))
        outputs = self.hidden_to_output(h)

        return outputs


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
