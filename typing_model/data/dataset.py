from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import gc
import torch

class TypingDataSet(Dataset):

    def __init__(self, mention, left_side, right_side, label, id2label, label2id, vocab_size):
        self.mention = mention
        self.left_side = left_side
        self.right_side = right_side
        self.label = label
        self.vocab_size = vocab_size
        self.id2label = id2label
        self.label2id = label2id
        self.label_id = [[self.label2id[v] for v in k] for k in self.label]

    def __len__(self):
        return len(self.left_side)

    def __getitem__(self, idx):
        labels_id = self.label_id[idx]
        one_hot = torch.zeros()
        one_hot = one_hot[labels_id]

        return self.mention[idx], self.left_side[idx], self.right_side[idx], one_hot


class TypingBERTDataSet(TypingDataSet):

    def __init__(self, mention, left_side, right_side, label, id2label, label2id, vocab_size):
        super().__init__(mention, left_side, right_side, label, id2label, label2id, vocab_size)
        model = SentenceTransformer('bert-large-uncased')

        self.mentions = model.encode(mention, show_progress_bar=True, batch_size=100)
        self.left_side = model.encode(left_side, show_progress_bar=True, batch_size=100)
        self.right_side = model.encode(right_side, show_progress_bar=True, batch_size=100)

        del model
        torch.cuda.empty_cache()
        gc.collect()


