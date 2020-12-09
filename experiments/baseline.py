from typing_model.models.baseline import *
from typing_model.data.parse_dataset import DatasetParser
from typing_model.data.dataset import TypingBERTDataSet
from torch.utils.data import DataLoader

pt_train = DatasetParser("../data/test_data.json")
id2label, label2id, vocab_len = pt_train.collect_global_config()
mention_train, left_side_train, right_side_train, label_train = pt_train.parse_dataset()

pt_test = DatasetParser("../data/test_data.json")
mention_test, left_side_test, right_side_test, label_test = pt_test.parse_dataset()

dataset_train = TypingBERTDataSet(mention_train, left_side_train, right_side_train, label_train,
                                  id2label, label2id, vocab_len)

dataloader_train = DataLoader(dataset_train)

dataset_test = TypingBERTDataSet(mention_test, left_side_test, right_side_test, label_test, id2label,
                                 label2id, vocab_len)

dataloader_test = DataLoader(dataset_test)

bt = BaseBERTTyper(vocab_len)

bt.train_(dataloader_train, dataloader_test)
