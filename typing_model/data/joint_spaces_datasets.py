from typing_model.data.BERT_datasets import ConcatenatedContextTypingBERTDataSet
from random import choice
import torch

class RandomSingleLabelDataset(ConcatenatedContextTypingBERTDataSet):

    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size, mention_max_tokens, context_max_tokens):
        super().__init__(mentions, left_side, right_side, label, id2label, label2id, vocab_size, mention_max_tokens, context_max_tokens)

    def __getitem__(self, idx):
        labels_id = self.label_id[idx]
        one_hot = torch.zeros(self.vocab_size)
        one_hot[labels_id] = 1
        # id = choice(labels_id)
        # random_correct_label = torch.zeros(self.vocab_size)
        # random_correct_label[id] = 1

        return self.mentions[idx], self.contexts[idx], one_hot