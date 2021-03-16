from os.path import join
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.dropout import Dropout
from typing_model.models.BERT_models import ConcatenatedContextBERTTyper
from pytorch_metric_learning.distances import CosineSimilarity
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.parameter import Parameter
import torch
from torch.nn.functional import normalize
from torch.nn import Embedding, Linear, Module


class ProjectionNetwork(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear_1 = Linear(in_dim, int(in_dim / 2))
        # self.linear_2 = Linear(int(in_dim / 2), int(in_dim / 4))
        # self.linear_3 = Linear(int(in_dim / 4), int(in_dim / 8))
        self.linear_4 = Linear(int(in_dim / 2), out_dim)

        self.relu = ReLU()
        self.droppy = Dropout(.2)
        self.bn1 = BatchNorm1d(int(in_dim / 2))
        # self.bn2 = BatchNorm1d(int(in_dim / 4))
        # self.bn3 = BatchNorm1d(int(in_dim / 8))

    def forward(self, inp):

        h = self.droppy(self.relu(self.bn1(self.linear_1(inp))))
        # h = self.droppy(self.relu(self.bn2(self.linear_2(h))))
        # h = self.droppy(self.relu(self.bn3(self.linear_3(h))))
        h = self.linear_4(h)

        return h



class BaseJointSpaceModel(ConcatenatedContextBERTTyper):
    def __init__(self, classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, 
                    loss_multiplier, positive_weight, joint_space_dim=100, trainable_threshold = False, random_threshold = False,
                    threshold_value=.5, inject_inference_in_model = False, regularization = False):
        super().__init__(classes, id2label, label2id, max_mention_size=max_mention_size, max_context_size=max_context_size, 
                            weights=weights, lr=lr, bert_fine_tuning=bert_fine_tuning, loss_multiplier=loss_multiplier, 
                            positive_weight=positive_weight)

        # self.input_to_hidden = Linear(400, 50)
        # self.input_to_joint_space = Linear(400, joint_space_dim)
        self.proj_net = ProjectionNetwork(400, joint_space_dim).cuda()
        self.pairwise_cosine = CosineSimilarity()
        self.sig = Sigmoid()
        self.inject_inference_in_model = inject_inference_in_model
        self.beta = 0.1
        self.regularization = regularization

        if trainable_threshold:
            if random_threshold:
                self.inference_threshold = Parameter(torch.rand(classes), requires_grad=True)
            else:
                self.inference_threshold = Parameter(torch.full((classes,), fill_value = threshold_value, requires_grad=True))
        else:
            self.inference_threshold = Parameter(torch.full((classes,), fill_value = threshold_value), requires_grad = False)

    def forward(self, mention, context, avoid_injected_inference = False):
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

        # h = self.droppy(self.relu(self.input_to_hidden(h)))
        joint_space_input_embedding = self.proj_net(h)

        preds = self.compute_vector_similarity(joint_space_input_embedding, self.get_label_embedding())
        
        if self.inject_inference_in_model and not avoid_injected_inference:
            preds = self.injectInference(preds)

        return preds
    
    def forward_return_input_proj(self, mention, context):
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

        # h = self.droppy(self.relu(self.input_to_hidden(h)))
        joint_space_input_embedding = self.proj_net(h)
        return joint_space_input_embedding

    def get_label_embedding(self):
        return self.label_embedding.weight

    def injectInference(self, preds):
        # preds range is [-1, 1], BCE domain is [0, 1], each pred over the respective inference_threshold will be inferred
        # 1 + inference_thresholds works as an offset on [-1, 1] if I want to use a threshold I am implicitly saying that
        # each value > threshold will be treated as correct; when performing preds * (1 / inference_threshold)
        # if preds > inference_threshold then preds * (1 / inference_treshold) > 1; so with this transformation the 
        # inference value is INJECTED in the model 
        # return torch.clamp(
        #             torch.clamp(preds, min = 0., max = 1.) * (1 / torch.clamp(self.inference_threshold + self.beta, 
        #                                                                 min=0., 
        #                                                                 max = 1)),
        #             min = 0., max = 1.)
        one = torch.ones(preds.shape).cuda()
        zero = torch.zeros(preds.shape).cuda()

        preds_q = preds ** 4

        return torch.where(preds >= self.inference_threshold, one, torch.where(preds > 0, preds_q, zero))

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
        
        model_output_wo_inference = self(mention_x, contexts_x, avoid_injected_inference=True)

        self.log_(val_loss=val_loss, model_output=model_output, model_output_wo_inference=model_output_wo_inference, labels=labels)

        self.update_metrics(pred=model_output_wo_inference, labels=labels)

        return val_loss
    
    def compute_loss(self, pred, true):
        if self.regularization:
            return self.get_loss_function(pred, true) + self.get_regularization()
        else:
            return self.get_loss_function(pred, true)

    def get_loss_function(self, pred, true):
        if self.inject_inference_in_model:
            true = true * torch.clamp(self.inference_threshold, min = 0., max = 1.)
        return self.apply_loss_weights(self.loss_function(pred, true))

    def get_regularization(self):
        return self.compute_label_cohesion(self.label_embedding.weight)

    def log_(self, val_loss, model_output, model_output_wo_inference, labels):
        self.log('losses/val_loss', val_loss, on_epoch=True, on_step=False)
        self.log('other_metrics/average_inference_value', torch.mean(self.get_inference_threshold()), on_epoch=True, on_step=False)
        self.log('other_metrics/inference_value_std', torch.std(self.get_inference_threshold()), on_epoch=True, on_step=False)
        self.log('other_metrics/labels_cohesion', torch.mean(self.compute_label_cohesion(self.label_embedding.weight)), 
                                                  on_epoch=True, on_step=False)

        self.log('logit_metrics/average_positive_logits', torch.mean(model_output[(labels * model_output_wo_inference).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)
        self.log('logit_metrics/average_negative_logits', torch.mean(model_output[((1 - labels) * model_output_wo_inference).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)

        self.log('logit_metrics/inferenced_average_positive_logits', torch.mean(model_output[(labels * model_output).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)
        self.log('logit_metrics/inferenced_average_negative_logits', torch.mean(model_output[((1 - labels) * model_output).nonzero(as_tuple=True)]), on_epoch=True, on_step=False)


    def get_inference_threshold(self):
        return self.inference_threshold

    def norm_int_prod(self, a, b):
        a = normalize(a, dim=1)
        b = normalize(b, dim=1)
        return torch.matmul(a, torch.transpose(b, 0, 1))

    def compute_vector_similarity(self, a, b):
        return self.norm_int_prod(a, b)
    
    def compute_label_cohesion(self, a):
        sim = self.norm_int_prod(a, a)
        classes = len(a)

        return (torch.sum(sim) - classes)/(classes**2 - classes)

        
    def get_discrete_pred(self, pred):
        mask = pred >= self.inference_threshold

        ones = torch.ones(mask.shape).cuda()
        zeros = torch.zeros(mask.shape).cuda()

        discrete_pred = torch.where(mask, ones, zeros)

        max_values_and_indices = torch.max(pred, dim = 1)

        for dp, i in zip(discrete_pred, max_values_and_indices.indices):
            dp[i] = 1
        
        return discrete_pred

    def update_metrics(self, pred, labels):

        pred = self.get_discrete_pred(pred)
        self.micro_f1.update(preds=pred, target=labels)
        self.micro_precision.update(preds=pred, target=labels)
        self.micro_recall.update(preds=pred, target=labels)

        self.macro_f1.update(preds=pred, target=labels)
        self.macro_precision.update(preds=pred, target=labels)
        self.macro_recall.update(preds=pred, target=labels)

        self.my_metrics.update(preds=pred, target=labels)

class ExplicitClassifier(BaseJointSpaceModel):

    def __init__(self, classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim, trainable_threshold, random_threshold, threshold_value, inject_inference_in_model, regularization):
        super().__init__(classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim=joint_space_dim, trainable_threshold=trainable_threshold, random_threshold=random_threshold, threshold_value=threshold_value, inject_inference_in_model=inject_inference_in_model, regularization = regularization)
        self.label_embedding = Embedding(num_embeddings=classes, embedding_dim=classes, max_norm=1)
        self.label_embedding.weight = torch.nn.Parameter(torch.diag(torch.ones(classes)), requires_grad = False)
        self.proj_net = ProjectionNetwork(400, classes).cuda()

    def compute_vector_similarity(self, a, b):
        # since b is the identity matrix and a is the result of a sigmoid, each dimension will be between 0 and 1
        a = self.sig(a)
        return torch.matmul(a, torch.transpose(b, 0, 1))

class JointSpaceModel(BaseJointSpaceModel):
    def __init__(self, classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim, trainable_threshold, random_threshold, threshold_value, inject_inference_in_model, regularization):
        super().__init__(classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim=joint_space_dim, trainable_threshold=trainable_threshold, random_threshold=random_threshold, threshold_value=threshold_value, inject_inference_in_model=inject_inference_in_model, regularization = regularization)

class RandomInitializedJointModel(JointSpaceModel):
    def __init__(self, classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim, trainable_threshold, random_threshold, threshold_value, inject_inference_in_model, regularization):
        super().__init__(classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim=joint_space_dim, trainable_threshold=trainable_threshold, random_threshold=random_threshold, threshold_value=threshold_value, inject_inference_in_model=inject_inference_in_model, regularization = regularization)
        self.label_embedding = Embedding(num_embeddings=classes, embedding_dim=joint_space_dim, max_norm=1)

class PartialInitializedJointModel(JointSpaceModel):
    def __init__(self, classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim, trainable_threshold, random_threshold, threshold_value, inject_inference_in_model, regularization, label_init_path):
        super().__init__(classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim, trainable_threshold, random_threshold, threshold_value, inject_inference_in_model, regularization=regularization)

        self.label_init_path = label_init_path
        initialized_labels = self.get_partial_initialization()
        self.label_embedding = Linear(joint_space_dim, classes - initialized_labels)
    
    def get_partial_initialization(self):
        with open(self.label_init_path, 'r') as inp:
            lines = [l.replace('\n', '') for l in inp.readlines()]
            lines = [l.split('\t') for l in lines]
        
        init_dict = {l[0]: [float(l_) for l_ in l[1:]] for l in lines}

        self.static_label_embedding = Parameter(torch.tensor([v for v in init_dict.values()]), requires_grad = False).cuda()

        self.modify_dicts(init_dict)
        
        return len(init_dict)
    
    def modify_dicts(self, init_dict):
        new_label2id = {k:i for i, k in enumerate(init_dict.keys())}

        for key in self.label2id:
            if key not in new_label2id:
                new_label2id[key] = len(new_label2id)
        new_id2label = {v:k for k,v in new_label2id.items()}

        self.old_label2id = self.label2id
        self.old_id2label = self.id2label

        self.label2id = new_label2id
        self.id2label = new_id2label
        self.setup_permutation_matrix()
    
    def setup_permutation_matrix(self):
        class_number = len(self.label2id)
        permutation_matrix = torch.zeros((class_number, class_number))

        for i in range(class_number):
            permutation_matrix[i][self.old_label2id[self.id2label[i]]] = 1
        self.permutation_matrix = Parameter(permutation_matrix, requires_grad=False)
        
    def traduce_labels(self, labels):
        return torch.matmul(labels, torch.transpose(self.permutation_matrix, 1, 0))

    def training_step(self, batch, batch_step):
        mention_x, contexts_x, labels = batch

        labels = self.traduce_labels(labels)

        model_output = self(mention_x, contexts_x)

        loss = self.compute_loss(model_output, labels)

        self.log('losses/train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_step):
        mention_x, contexts_x, labels = batch

        labels = self.traduce_labels(labels)

        model_output = self(mention_x, contexts_x)

        val_loss = self.compute_loss(model_output, labels)
        
        model_output_wo_inference = self(mention_x, contexts_x, avoid_injected_inference=True)

        self.log_(val_loss=val_loss, model_output=model_output_wo_inference, labels=labels)

        self.update_metrics(pred=model_output_wo_inference, labels=labels)

        return val_loss
    
    def get_label_embedding(self):

        return torch.cat((self.static_label_embedding, self.label_embedding.weight), dim = 0)

class TrainableOneHotModel(JointSpaceModel):
    def __init__(self, classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim, trainable_threshold, random_threshold, threshold_value, inject_inference_in_model, regularization):
        super().__init__(classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim=joint_space_dim, trainable_threshold=trainable_threshold, random_threshold=random_threshold, threshold_value=threshold_value, inject_inference_in_model=inject_inference_in_model, regularization = regularization)
        self.label_embedding = Embedding(num_embeddings=classes, embedding_dim=classes, max_norm=1)
        self.label_embedding.weight = torch.nn.Parameter(torch.diag(torch.ones(classes)))

class OneHotInjectedInferenceJointModel(JointSpaceModel):
    def __init__(self, classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim, trainable_threshold, random_threshold, threshold_value, inject_inference_in_model, regularization):
        super().__init__(classes, id2label, label2id, max_mention_size, max_context_size, weights, lr, bert_fine_tuning, loss_multiplier, positive_weight, joint_space_dim=joint_space_dim, trainable_threshold=trainable_threshold, random_threshold=random_threshold, threshold_value=threshold_value, inject_inference_in_model=inject_inference_in_model, regularization = regularization)
        
        self.label_embedding = Embedding(num_embeddings=classes, embedding_dim=classes, max_norm=1)
        self.label_embedding.weight = torch.nn.Parameter(torch.diag(torch.ones(classes)), requires_grad = False)