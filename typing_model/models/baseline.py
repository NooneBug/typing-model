from torch import nn
import torch
from tqdm import tqdm


class BaseBERTTyper(nn.Module):

    def __init__(self, classes):
        super().__init__()

        self.mention_to_hidden = nn.Linear(1024, 200)
        self.left_to_hidden = nn.Linear(1024, 200)
        self.right_to_hidden = nn.Linear(1024, 200)

        self.hidden_to_output = nn.Linear(600, classes)

        self.classification_loss = nn.BCEWithLogitsLoss()
        self.epochs = 5

        self.early_stopping = True
        self.patience = 3
        self.early_stopping_trigger = False

    def forward(self, mention, left, right):

        h1 = self.mention_to_hidden(mention)
        h2 = self.left_to_hidden(left)
        h3 = self.right_to_hidden(right)
        concat = torch.cat([h1, h2, h3], dim=1)

        outputs = self.hidden_to_output(concat)

        return outputs

    def train_(self, train_loader, val_loader):

        loss_SUM = 0
        val_loss_SUM = 0
        total_examples = 0
        total_val_examples = 0

        e = 0
        while e < self.epochs and not self.early_stopping_trigger:
            pbar = tqdm(total=len(train_loader), desc='{}^ epoch: training'.format(e + 1))
            # TODO: rename data and split in 4 variables
            for data in train_loader:
                self.optimizer.zero_grad()
                self.train()

                model_output = self(data[0], data[1], data[2])

                # print('model_output: {}'.format(model_output))

                labels = data[3]

                loss = self.compute_loss(model_output, labels)

                loss.backward()

                loss_SUM += loss.item()
                total_examples += len(data)

                pbar.update(1)

                self.optimizer.step()
            pbar.close()

            with torch.no_grad():
                self.eval()

                pbar = tqdm(total=len(val_loader), desc='{}^ epoch: validation'.format(e + 1))
                # TODO: rename data and split in 4 variables
                for data in val_loader:

                    model_output = self(data[0], data[1], data[2])

                    labels = data[3]
                    val_loss = self.compute_loss(model_output, labels)

                    val_loss_SUM += val_loss
                    total_val_examples += len(data)
                    pbar.update(1)

                pbar.close()

            self.early_stopping_routine(value=val_loss_SUM, epoch=e)
            e += 1

    def compute_loss(self, pred, true):
        return self.classification_loss(pred, true)

    def early_stopping_routine(self, value, epoch):
        if self.early_stopping:
            if epoch == 0:
                self.min_val_loss = value
                self.best_epoch = epoch
                self.save_model(epoch)
            elif value <= self.min_val_loss:
                self.best_epoch = epoch
                self.save_model(epoch)
            elif self.best_epoch + self.patience < epoch:
                print('EarlyStopping')
                self.early_stopping_trigger = True
        print('\t best epoch: {}\n'.format(self.best_epoch))
    
    def save_model(self, epoch):
        # TO DO: update the path where save the model  
        torch.save({
                'model_state_dict' : self.state_dict(),
                'epoch' : epoch 
              }, 
              '/model.pth')