from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.loss import BCELoss
from torch.nn.parameter import Parameter
from typing_model.models.BERT_models import ConcatenatedContextBERTTyper
from torch.nn import Embedding, Linear
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from torch.nn.functional import normalize

class JointSpaceModel(ConcatenatedContextBERTTyper):

    def __init__(self, classes, id2label, label2id, max_mention_size=10, max_context_size=16, weights=None, lr=1e-3, bert_fine_tuning=None, loss_multiplier=1, positive_weight = 1, joint_space_dim=100):
        super().__init__(classes, id2label, label2id, max_mention_size=max_mention_size, max_context_size=max_context_size, weights=weights, lr=lr, bert_fine_tuning=bert_fine_tuning, loss_multiplier=loss_multiplier, positive_weight = positive_weight)

        self.label_embedding = Embedding(num_embeddings=classes, embedding_dim=classes, max_norm=1)

        self.input_to_joint_space = Linear(400, joint_space_dim)

        self.pairwise_cosine = CosineSimilarity()

        self.sig = Sigmoid()

        self.classification_loss = torch.nn.BCELoss()

        self.margins = Parameter(torch.randn(classes))

    def forward(self, mention, context):

        # Parent forward
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

        #end parent forward

        joint_space_input_embedding = self.input_to_joint_space(h)

        # joint_space_input_embedding = normalize(joint_space_input_embedding, dim = 1)
        # norm_labels = normalize(self.label_embedding.weight, dim = 1)

        preds = torch.clamp(self.compute_vector_similarity(joint_space_input_embedding, self.label_embedding.weight), min = 0., max = 1.)

        return preds

    def compute_vector_similarity(self, a, b):
        a = normalize(a, dim = 1)
        b = normalize(b, dim=1)
        return torch.matmul(a, torch.transpose(b, 0, 1))

    def training_step(self, batch, batch_step):
        mention_x, contexts_x, labels = batch

        labels = labels.cuda()

        model_output = self(mention_x, contexts_x)

        loss = self.compute_loss(model_output, labels)

        self.log('losses/train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def get_margin(self):
        return torch.tensor([0.])

    def validation_step(self, batch, batch_step):
        mention_x, contexts_x, labels = batch

        labels = labels.cuda()

        model_output = self(mention_x, contexts_x)

        val_loss = self.compute_loss(model_output, labels)
        
        self.log('losses/val_loss', val_loss, on_epoch=True, on_step=False)
        self.log('other_metrics/average_margin', torch.mean(self.get_margin()), on_epoch=True, on_step=False)
        self.log('other_metrics/margins_std', torch.std(self.get_margin()), on_epoch=True, on_step=False)
        self.log('other_metrics/labels_cohesion', torch.mean(self.compute_label_cohesion(self.label_embedding.weight)), 
                                                  on_epoch=True, on_step=False)

        self.log('logit_metrics/average_positive_logits', torch.mean(model_output[(labels * model_output).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)
        self.log('logit_metrics/average_negative_logits', torch.mean(model_output[((1 - labels) * model_output).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)

        # non_reduced_loss = BCELoss(reduction='none')
        # loss_ = non_reduced_loss(model_output, labels)

        # self.log('logit_metrics/average_positive_loss_values', torch.mean(loss_[(labels * loss_).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)
        # self.log('logit_metrics/average_negative_loss_values', torch.mean(loss_[((1 - labels) * loss_).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)

        self.update_metrics(pred=model_output, labels=labels)

        return val_loss

    def norm_int_prod(self, a, b):
        a = normalize(a, dim=1)
        b = normalize(b, dim=1)
        return torch.matmul(a, torch.transpose(b, 0, 1))

    def compute_label_cohesion(self, a):

        sim = self.norm_int_prod(a, a)
        classes = len(a)

        return (torch.sum(sim) - classes)/(classes**2 - classes)

    def update_metrics(self, pred, labels):

        pred = self.get_discrete_pred(pred, threshold=.5)

        self.micro_f1.update(preds=pred, target=labels)
        self.micro_precision.update(preds=pred, target=labels)
        self.micro_recall.update(preds=pred, target=labels)

        self.macro_f1.update(preds=pred, target=labels)
        self.macro_precision.update(preds=pred, target=labels)
        self.macro_recall.update(preds=pred, target=labels)

        self.my_metrics.update(preds=pred, target=labels)

class RegularizedJointModel(JointSpaceModel):
    def __init__(self, classes, id2label, label2id, max_mention_size=10, max_context_size=16, weights=None, lr=1e-3, 
                    bert_fine_tuning=None, loss_multiplier=1, positive_weight = 1, joint_space_dim=100):
        super().__init__(classes, id2label, label2id, max_mention_size=max_mention_size, 
                            max_context_size=max_context_size, weights=weights, lr=lr, bert_fine_tuning=bert_fine_tuning, 
                            loss_multiplier=loss_multiplier, positive_weight=positive_weight, joint_space_dim=joint_space_dim)

    def loss_function(self, pred, true):
        return self.classification_loss(pred, true) + torch.mean(self.compute_vector_similarity(self.label_embedding.weight, 
                                                                                                self.label_embedding.weight))

class MarginJointModel(JointSpaceModel):
    def __init__(self, classes, id2label, label2id, max_mention_size=10, max_context_size=16, weights=None, lr=1e-3, bert_fine_tuning=None, loss_multiplier=1, positive_weight = 1, joint_space_dim=100):
        super().__init__(classes, id2label, label2id, max_mention_size=max_mention_size, 
                            max_context_size=max_context_size, weights=weights, lr=lr, bert_fine_tuning=bert_fine_tuning, 
                            loss_multiplier=loss_multiplier, positive_weight= positive_weight, joint_space_dim=joint_space_dim)

        self.margin_parameters = Parameter(torch.randn(classes))
    
    def forward(self, mention, context):
        preds = super().forward(mention=mention, context=context)

        preds = torch.clamp(preds + self.get_margin(), min = 0., max = 1.)

        return preds
    
    def get_margin(self):
        return self.sig(self.margins) / 2

class MarginRegJointModel(JointSpaceModel):
    def __init__(self, classes, id2label, label2id, max_mention_size=10, max_context_size=16, weights=None, lr=1e-3, bert_fine_tuning=None, loss_multiplier=1, joint_space_dim=100, positive_weight = 1):
        super().__init__(classes, id2label, label2id, max_mention_size=max_mention_size, 
                            max_context_size=max_context_size, weights=weights, lr=lr, bert_fine_tuning=bert_fine_tuning, 
                            loss_multiplier=loss_multiplier, positive_weight = positive_weight, joint_space_dim=joint_space_dim)

        self.margin_parameters = Parameter(torch.randn(classes))
    
    def forward(self, mention, context):
        preds = super().forward(mention=mention, context=context)

        preds = torch.clamp(preds + self.get_margin(), min = 0., max = 1.)

        return preds
    
    def get_margin(self):
        return self.sig(self.margins) / 2

    def loss_function(self, pred, true):
        return self.classification_loss(pred, true) + torch.mean(self.compute_label_cohesion(self.label_embedding.weight))


class oneHotJointModel(ConcatenatedContextBERTTyper):
    def __init__(self, classes, id2label, label2id, max_mention_size=10, max_context_size=16, weights=None, lr=1e-3, bert_fine_tuning=None, loss_multiplier=1, positive_weight = 1):
        super().__init__(classes, id2label, label2id, max_mention_size=max_mention_size, max_context_size=max_context_size, weights=weights, lr=lr, bert_fine_tuning=bert_fine_tuning, loss_multiplier=loss_multiplier, positive_weight = positive_weight)

        self.input_to_joint_space = Linear(400, classes)
        self.label_embedding = Parameter(torch.diag(torch.ones(classes)), requires_grad=False)
        self.classification_loss = torch.nn.BCELoss()

    def forward(self, mention, context):

        # Parent forward
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

        #end parent forward

        joint_space_input_embedding = self.input_to_joint_space(h)

        preds = self.compute_vector_similarity(self.sig(joint_space_input_embedding), self.label_embedding)

        return preds

    def compute_vector_similarity(self, a, b):
        return torch.matmul(a, b)

    def training_step(self, batch, batch_step):
        mention_x, contexts_x, labels = batch

        labels = labels.cuda()

        model_output = self(mention_x, contexts_x)

        loss = self.compute_loss(model_output, labels)

        self.log('losses/train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_step):
        mention_x, contexts_x, labels = batch

        labels = labels.cuda()

        model_output = self(mention_x, contexts_x)

        val_loss = self.compute_loss(model_output, labels)
        
        self.log('losses/val_loss', val_loss, on_epoch=True, on_step=False)
        self.log('other_metrics/labels_cohesion', torch.mean(self.compute_label_cohesion(self.label_embedding)), 
                                                  on_epoch=True, on_step=False)
        self.log('logit_metrics/average_positive_logits', torch.mean(model_output[(labels * model_output).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)
        self.log('logit_metrics/average_negative_logits', torch.mean(model_output[((1 - labels) * model_output).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)

        # non_reduced_loss = BCELoss(reduction='none')
        # loss_ = non_reduced_loss(model_output, labels)

        # self.log('logit_metrics/average_positive_loss_values', torch.mean(loss_[(labels * loss_).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)
        # self.log('logit_metrics/average_negative_loss_values', torch.mean(loss_[((1 - labels) * loss_).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)

        self.update_metrics(pred=model_output, labels=labels)

        return val_loss

    def norm_int_prod(self, a, b):
        a = normalize(a, dim=1)
        b = normalize(b, dim=1)
        return torch.matmul(a, torch.transpose(b, 0, 1))

    def compute_label_cohesion(self, a):

        sim = self.norm_int_prod(a, a)
        classes = len(a)
        

        return (torch.sum(sim) - classes)/(classes**2 - classes)

    def update_metrics(self, pred, labels):

        pred = self.get_discrete_pred(pred)

        self.micro_f1.update(preds=pred, target=labels)
        self.micro_precision.update(preds=pred, target=labels)
        self.micro_recall.update(preds=pred, target=labels)

        self.macro_f1.update(preds=pred, target=labels)
        self.macro_precision.update(preds=pred, target=labels)
        self.macro_recall.update(preds=pred, target=labels)

        self.my_metrics.update(preds=pred, target=labels)

class trainableOneHotJointModel(oneHotJointModel):
    def __init__(self, classes, id2label, label2id, max_mention_size=10, max_context_size=16, weights=None, lr=1e-3, bert_fine_tuning=None, loss_multiplier=1, positive_weight = 1, joint_space_dim = 100):
        super().__init__(classes, id2label, label2id, max_mention_size=max_mention_size, max_context_size=max_context_size, weights=weights, lr=lr, bert_fine_tuning=bert_fine_tuning, loss_multiplier=loss_multiplier, positive_weight=positive_weight)

        self.input_to_joint_space = Linear(400, joint_space_dim)
        self.label_embedding = Parameter(torch.diag(torch.ones(classes)))
        self.classification_loss = torch.nn.BCELoss()
        self.pairwise_cosine = CosineSimilarity()


        if self.label_embedding.shape[0] < joint_space_dim:
            self.add_zeros_to_joint_space(joint_space_dim)
        elif self.label_embedding.shape[0] > joint_space_dim:
            raise Exception('Please provide a joint_spaced_dimension ({}) higher than classes number ({})'.format(joint_space_dim, classes))
  
    def add_zeros_to_joint_space(self, joint_space_dim):

        to_add = joint_space_dim - self.label_embedding.shape[1]

        new_tensors = torch.tensor([list(t.detach().cpu()) + [0. for _ in range(to_add)] for t in self.label_embedding])

        self.label_embedding = Parameter(new_tensors, requires_grad=True)
      
    def forward(self, mention, context):

        # Parent forward
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

        #end parent forward

        joint_space_input_embedding = self.input_to_joint_space(h)

        preds = torch.clamp(self.compute_vector_similarity(joint_space_input_embedding, self.label_embedding), min = 0., max = 1.)

        return preds

    def compute_vector_similarity(self, a, b):
        # return (1 + self.pairwise_cosine(a, b))/2
        a = normalize(a, dim = 1)
        b = normalize(b, dim = 1)
        return torch.matmul(a, torch.transpose(b, 0, 1))