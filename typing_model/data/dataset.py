from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import gc
import torch

class TypingDataBase(Dataset):

    def __init__(self, mention, left_side, right_side, label):
        self.mention = mention
        self.left_side = left_side
        self.right_side = right_side
        self.label = label

    def __len__(self):
        return len(self.left_side)

    def __getitem__(self, idx):
        return self.mention[idx], self.left_side[idx], self.right_side, self.label


class TypingBERTDataBase(TypingDataBase):

    def __init__(self, mention, left_side, right_side, label):
        super().__init__(mention, left_side, right_side, label)
        model = SentenceTransformer('bert-large-uncased')

        self.mentions = model.encode(mention, show_progress_bar=True, batch_size=100)
        self.left_side = model.encode(left_side, show_progress_bar=True, batch_size=100)
        self.right_side = model.encode(right_side, show_progress_bar=True, batch_size=100)
        self.labels = label
        del model
        torch.cuda.empty_cache()
        gc.collect()


