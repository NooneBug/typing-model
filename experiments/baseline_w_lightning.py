from typing_model.models.baseline import BaseBERTTyper
from typing_model.data.parse_dataset import DatasetParser
from typing_model.data.dataset import TypingBERTDataSet
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_dataloader_from_dataset_path(dataset_path, batch_size = 500, shuffle = False, train = False, id2label = None, label2id = None, vocab_len = None):
    pt = DatasetParser(dataset_path)

    if train:
        id2label, label2id, vocab_len = pt.collect_global_config()
    elif not id2label or not label2id or not vocab_len:
        raise Exception('Please provide id2label_dict, label2id_dict and vocab len to generate val_loader or test_loader')
    
    mention, left_side, right_side, label = pt.parse_dataset()

    dataset = TypingBERTDataSet(mention, left_side, right_side, label, id2label, label2id, vocab_len)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle, num_workers=10)

    if not train:
        return dataloader
    else:
        return dataloader, id2label, label2id, vocab_len


# trainset_path = "/datahdd/vmanuel/entity_typing_all_datasets/data/ontonotes/g_train_tree.json"
trainset_path = "../data/100_x_type.json"
valset_path = "/datahdd/vmanuel/entity_typing_all_datasets/data/ontonotes/g_dev_tree.json"
testset_path = "../../data/test_1k.json"

dataloader_train, id2label, label2id, vocab_len = get_dataloader_from_dataset_path(trainset_path, 
                                                                                    shuffle=True, train = True)

dataloader_val = get_dataloader_from_dataset_path(valset_path,
                                                    id2label=id2label, label2id=label2id, vocab_len=vocab_len)

bt = BaseBERTTyper(vocab_len, id2label, label2id)

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=10,
   verbose=False,
   mode='min'
)

trainer = Trainer(callbacks=[early_stop_callback])

trainer.fit(bt, dataloader_train, dataloader_val)