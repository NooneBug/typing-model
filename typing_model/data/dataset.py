from torch.utils.data import Dataset

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
