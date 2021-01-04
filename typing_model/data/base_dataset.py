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
        self.label_id = [[self.label2id[v] for v in k] for k in self.label]

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        labels_id = self.label_id[idx]
        one_hot = torch.zeros(self.vocab_size)
        one_hot[labels_id] = 1

        return self.mentions[idx], self.left_side[idx], self.right_side[idx], one_hot
