from torch.utils.data import Dataset
from typing_model.data.utils import TreeMaker


class KermitDataset(Dataset):

    # ON HOLD: waiting for issue

    def __init__(self, sentences):
        self.sentences = sentences

        tm = TreeMaker()
        parsed = [tm.parse(k) for k in self.sentences]


    def __len__(self):
        return len(self.sentences)