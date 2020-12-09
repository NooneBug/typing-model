from typing_model.models.baseline import *
from typing_model.data.parse_dataset import DatasetParser
from typing_model.data.dataset import TypingBERTDataSet
from torch.utils.data import DataLoader

pt_train = DatasetParser("../data/g_dev_tree.json")
parsed_train = pt_train.parse_dataset()

dataset_train = TypingBERTDataSet(*parsed_train)
dataloader_train = DataLoader(dataset_train)

pt_test = DatasetParser("../data/g_dev_tree.json")
parsed_test = pt_test.parse_dataset()

dataset_test = TypingBERTDataSet(*parsed_test)
dataloader_test = DataLoader(dataset_test)

bt = BaseBERTTyper(89)

bt.train_(dataloader_train, dataloader_test)
