from typing_model.models.baseline import *
from typing_model.data.parse_dataset import DatasetParser
from typing_model.data.dataset import TypingBERTDataBase
from torch.utils.data import DataLoader

pt = DatasetParser("examples/toy_dataset.json")
parsed = pt.parse_dataset()

dataset = TypingBERTDataBase(*parsed)
dataloader = DataLoader(dataset)

bt = BaseBERTTyper()

bt(dataloader)
