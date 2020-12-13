import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class BaseTyper(nn.Module):
    def __init__(self, id2label, label2id, name = 'BaseTyper'):
        super().__init__()
        self.epochs = 50
        self.name = name

        self.early_stopping = True
        self.patience = 10
        self.early_stopping_trigger = False
        
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.id2label = id2label
        self.label2id = label2id

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.log_file = '/home/vmanuel/typing_network/typing-model/experiments/logs/' + self.name + '.txt'
        self.log_init()

    def train_(self, train_loader, val_loader):

        e = 0
        while e < self.epochs and not self.early_stopping_trigger:
            loss_SUM = 0
            val_loss_SUM = 0
            total_examples = 0
            total_val_examples = 0
            pbar = tqdm(total=len(train_loader), desc='{}^ epoch: training'.format(e + 1))
            for mention_x, left_x, right_x, labels in train_loader:
                self.optimizer.zero_grad()
                self.train()

                model_output = self(mention_x, left_x, right_x)

                loss = self.compute_loss(model_output, labels)

                loss.backward()

                loss_SUM += loss.item()
                total_examples += len(mention_x)

                pbar.update(1)

                self.optimizer.step()
            pbar.close()

            with torch.no_grad():
                self.eval()
                
                all_outputs =[]
                all_labels = []
                epoch_val_loss = 0

                pbar = tqdm(total=len(val_loader), desc='{}^ epoch: validation'.format(e + 1))
                for mention_x, left_x, right_x, labels in val_loader:

                    model_output = self(mention_x, left_x, right_x)

                    val_loss = self.compute_loss(model_output, labels)

                    epoch_val_loss += val_loss
                    total_val_examples += len(mention_x)
                    pbar.update(1)

                    all_outputs.append(self.sig(model_output).detach().cpu().numpy())
                    all_labels.append(labels.detach().cpu().numpy())

                val_loss_SUM += epoch_val_loss

                pbar.close()

            
            e += 1
            self.early_stopping_routine(value=epoch_val_loss, epoch=e)
            self.compute_and_log_metrics(epoch = e, val_loss = epoch_val_loss, model_outputs=all_outputs, batched_labels=all_labels)

    def logits_and_one_hot_labels_to_string(self, logits, one_hot_labels, no_void = False, threshold = 0.5):

        pred_classes, true_classes = [], []

        for example_logits, example_labels in zip(logits, one_hot_labels):
            mask = example_logits > threshold
            if no_void:
                argmax = np.argmax(example_logits)
                pc = self.id2label[argmax]
                p_classes = [self.id2label[i] for i, m in enumerate(mask) if m]

                if pc not in p_classes:
                    p_classes.append(pc)
                pred_classes.append(p_classes)

            else:    
                pred_classes.append([self.id2label[i] for i, m in enumerate(mask) if m])
            true_classes.append([self.id2label[i] for i, l in enumerate(example_labels) if l])
        
        assert len(pred_classes) == len(true_classes), "Error in id2label traduction"
        return pred_classes, true_classes


    def compute_f1(self, p, r):
        return (2*p*r)/(p + r) if (p + r) else 0


    def compute_metrics(self, pred_classes, true_classes):
        correct_counter = 0
        prediction_counter = 0
        true_labels_counter = 0
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0

        void_prediction_counter = 0

        for example_pred, example_true in zip(pred_classes, true_classes):
            
            assert len(example_true) > 0, 'Error in true label traduction'

            prediction_counter += len(example_pred)
            
            true_labels_counter += len(example_true)
            if not example_pred:
                void_prediction_counter += 1
            else:
                correct_predictions = len(set(example_pred).intersection(set(example_true)))
                correct_counter += correct_predictions
                
                p = correct_predictions / len(example_pred)
                r = correct_predictions / len(example_true)
                f1 = self.compute_f1(p, r)
                precision_sum += p
                recall_sum += r
                f1_sum += f1
                
            
        micro_p = correct_counter / prediction_counter
        micro_r = correct_counter / true_labels_counter
        micro_f1 = self.compute_f1(micro_p, micro_r)

        examples_in_dataset = len(true_classes)

        macro_p = precision_sum / examples_in_dataset
        macro_r = recall_sum / examples_in_dataset
        macro_f1 = f1_sum / examples_in_dataset

        avg_pred_number = prediction_counter / examples_in_dataset

        return avg_pred_number, void_prediction_counter, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1


    def log_init(self):
        out = open(self.log_file, 'a')
        out.write('{}\t{:4}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('epoch','average_predictions_number', 'void_predictions',
                                                                    'micro_p', 'micro_r', 'micro_f1', 
                                                                    'macro_p', 'macro_r', 'macro_f1',
                                                                    'eval_loss'))

    def log_epoch(self, epoch):
        out = open(self.log_file, 'a')

        out.write('{:4}'.format(epoch))

    def log_loss(self, loss):
        out = open(self.log_file, 'a')

        out.write('\t{:.4f}\n'.format(loss))


    def log_metrics(self,avg_pred_number, void_predictions, p, r, f1, ma_p, ma_r, ma_f1):
        out = open(self.log_file, 'a')
        
        out.write('\t{:4.2f}\t{:6}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(avg_pred_number,
                                                                                            void_predictions,
                                                                                            p, r, f1,
                                                                                            ma_p, ma_r, ma_f1))

    def log_divider(self):
        out = open(self.log_file, 'a')

        out.write('{:*^30}\n'.format(''))

    def compute_and_log_metrics(self, epoch, val_loss, model_outputs, batched_labels):
        self.log_epoch(epoch)
        pred_classes, true_classes = [], []
        for batch_output, batch_labels in zip(model_outputs, batched_labels):
            p, t = self.logits_and_one_hot_labels_to_string(logits=batch_output, one_hot_labels=batch_labels)
            pred_classes.extend(p)
            true_classes.extend(t)

        assert len(pred_classes) == len(true_classes), "Error in id2label traduction"

        avg_pred_number, void_predictions, p, r, f1, ma_p, ma_r, ma_f1 = self.compute_metrics(pred_classes=pred_classes,
                                                                                                true_classes=true_classes) 

        self.log_metrics(avg_pred_number, void_predictions, p, r, f1, ma_p, ma_r, ma_f1)
        self.log_loss(val_loss)

        pred_classes, true_classes = [], []
        for batch_output, batch_labels in zip(model_outputs, batched_labels):
            p, t = self.logits_and_one_hot_labels_to_string(logits=batch_output, one_hot_labels=batch_labels, no_void=True)
            pred_classes.extend(p)
            true_classes.extend(t)
        
        avg_pred_number, void_predictions, p, r, f1, ma_p, ma_r, ma_f1 = self.compute_metrics(pred_classes=pred_classes,
                                                                                                true_classes=true_classes) 
        self.log_epoch(epoch)
        self.log_metrics(avg_pred_number, void_predictions, p, r, f1, ma_p, ma_r, ma_f1)
        self.log_loss(val_loss)
        self.log_divider()


    def compute_loss(self, pred, true):
        return self.classification_loss(pred, true)

    def early_stopping_routine(self, value, epoch):
        if self.early_stopping:
            if epoch == 1:
                self.min_val_loss = value
                self.best_epoch = epoch
                self.save_model(epoch)
            elif value <= self.min_val_loss:
                self.min_val_loss = value
                self.best_epoch = epoch
                self.save_model(epoch)
            elif self.best_epoch + self.patience < epoch:
                print('EarlyStopping')
                self.early_stopping_trigger = True
        print('\t best epoch: {}\n'.format(self.best_epoch))

    def save_model(self, epoch):
        torch.save({
                'model_state_dict' : self.state_dict(),
                'epoch' : epoch
              },
              '/home/vmanuel/typing_network/typing-model/trained_models/{}.pth'.format(self.name))


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
