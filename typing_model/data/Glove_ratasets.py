from torch.utils.data import Dataset
from typing_model.data.utils import GloveManager

class GloveDataset(Dataset):
    def __init__(self, path_to_glove_embeddings):
        super().__init__()
        self.gm = GloveManager()
        embedding_dict = self.gm.load_embedding_dict(path_to_glove_embeddings)
        