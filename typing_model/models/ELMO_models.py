from typing_model.models.base_models import BaseTyper
from allennlp.modules.elmo import Elmo
from torch import nn
import torch

class ElmoTyper(BaseTyper):

    def __init__(self, classes, id2label, label2id, class_weights, option_file, elmo_weights, name = 'ElmoTyper'):
        super().__init__(classes, id2label, label2id, name=name, weights=class_weights)

        self.elmo_model = Elmo(options_file=option_file, 
                                weight_file=elmo_weights, 
                                num_output_representations=3, 
                                dropout=0).cuda()

        self.mention_to_hidden = nn.Linear(self.elmo_model.get_output_dim(), 200).cuda()
        self.context_to_hidden = nn.Linear(self.elmo_model.get_output_dim(), 200).cuda()

        self.hidden_to_class = nn.Linear(400, classes).cuda()

        self.droppy = nn.Dropout(0.2).cuda()
        self.relu = nn.ReLU().cuda()

        for param in self.elmo_model.parameters():
            param.requires_grad = False
    
    def forward(self, mention, context):
        mention = mention.cuda()
        context = context.cuda()

        mention_embed_layers = self.elmo_model(mention)['elmo_representations']
        mention_token_embed = torch.mean(torch.stack(mention_embed_layers), dim = 0)
        mention_embed = torch.mean(mention_token_embed, dim = 1)
        h1 = self.droppy(self.relu(self.mention_to_hidden(mention_embed)))

        context_embed_layers = self.elmo_model(context)['elmo_representations']
        context_token_embed = torch.mean(torch.stack(context_embed_layers), dim = 0)
        context_embed = torch.mean(context_token_embed, dim = 1)
        h2 = self.droppy(self.relu(self.context_to_hidden(context_embed)))

        hidden = torch.cat((h1, h2), dim = 1)

        output = self.hidden_to_class(hidden)

        return output

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
