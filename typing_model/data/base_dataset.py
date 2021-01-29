from torch.utils.data import Dataset
import torch

class TypingDataSet(Dataset):

    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size):
        self.mentions = mentions
        self.left_side = left_side
        self.right_side = right_side
        self.label = label
        self.vocab_size = vocab_size

        self.id2label = id2label
        self.label2id = label2id
        self.label_id = [[self.label2id[v] if v in self.label2id else 0 for v in k] for k in self.label]
        self.labels_not_in_training_set = 0

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        labels_id = self.label_id[idx]
        one_hot = torch.zeros(self.vocab_size)
        one_hot[labels_id] = 1

        return self.mentions[idx], self.left_side[idx], self.right_side[idx], one_hot

    def get_label_from_id(self, label):
        try:
            return self.label2id[label]
        except:
            # the id is not in the dictionary
            new_id = len(self.id2label)
            self.label2id[label] = new_id
            self.id2label[new_id] = label
            self.labels_not_in_training_set += 1
            self.vocab_size += 1
            return self.label2id[label]
    
    def print_out_of_train_labels_number(self):
        print('{} labels are in dev and not in train (on a total number of {}, train + dev)'.format(self.labels_not_in_training_set, 
                                                                                                    len(self.label2id)))