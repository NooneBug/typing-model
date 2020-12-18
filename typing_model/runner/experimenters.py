from build.lib.typing_model import data
from pytorch_lightning.utilities.cloud_io import load
from typing_model.data.parse_dataset import DatasetParser
from typing_model.data.dataset import TypingBERTDataSet
from torch.utils.data import DataLoader
from typing_model.models.baseline import BaseBERTTyper

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import configparser
import pickle
# from pytorch_lightning.loggers import TensorBoardLogger



class ExperimentRoutine():
    def __init__(self, exp_list, config_file):
        self.exp_list = exp_list
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def perform_experiments(self):
        for exp in self.exp_list:
            print('Performing experiment: {}'.format(exp['exp_name']))
            dt = exp['Dataclass'](**dict(self.config[exp['exp_name']]))
            exp = exp['ExperimentClass'](dataclass=dt)

            exp.setup()
            exp.perform_experiment()

class BaseExperimentClass():
    def setup(self, dataclass):
        # setup all class variables from dataclass
        raise NotImplementedError


    def perform_experiment(self):
        # perform the experiment
        raise NotImplementedError


class BertBaselineExperiment(BaseExperimentClass):

    def __init__(self, dataclass):
        self.train_data_path = dataclass.train_data_path
        self.eval_data_path = dataclass.eval_data_path
        self.test_data_path = dataclass.test_data_path
        
        self.early_stopping = dataclass.early_stopping
        self.early_stopping_patience = dataclass.early_stopping_patience
        self.epochs = dataclass.epochs

        self.load_train_dataset_path = dataclass.load_train_dataset_path
        self.load_eval_dataset_path = dataclass.load_eval_dataset_path
        self.load_test_dataset_path = dataclass.load_test_dataset_path
        
        self.save_train_dataset_path = dataclass.save_train_dataset_path
        self.save_eval_dataset_path = dataclass.save_eval_dataset_path
        self.save_test_dataset_path = dataclass.save_test_dataset_path

    def setup(self):
        self.dataloader_train, id2label, label2id, vocab_len = self.get_dataloader_from_dataset_path(self.train_data_path, 
                                                                                                shuffle=True, train = True,
                                                                                                load_path=self.load_train_dataset_path,
                                                                                                save_path=self.save_train_dataset_path)

        self.dataloader_val = self.get_dataloader_from_dataset_path(self.eval_data_path,
                                                                id2label=id2label, label2id=label2id, vocab_len=vocab_len,
                                                                load_path=self.load_eval_dataset_path,
                                                                save_path=self.save_eval_dataset_path)

        self.bt = BaseBERTTyper(vocab_len, id2label, label2id)

        # logger = TensorBoardLogger('tb_logs', name='my_model')

        if self.early_stopping:
            early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=self.early_stopping_patience,
            verbose=False,
            mode='min'
            )

            self.trainer = Trainer(callbacks=[early_stop_callback])
        else:
            self.trainer = Trainer()

    def perform_experiment(self):
        self.trainer.fit(self.bt, self.dataloader_train, self.dataloader_val)

    def get_dataloader_from_dataset_path(self, dataset_path, batch_size = 500, shuffle = False, train = False, load_path = None,
                                                save_path = None, id2label = None, label2id = None, vocab_len = None):
        
        pt = DatasetParser(dataset_path)

        if train:
            id2label, label2id, vocab_len = pt.collect_global_config()
        elif not id2label or not label2id or not vocab_len:
            raise Exception('Please provide id2label_dict, label2id_dict and vocab len to generate val_loader or test_loader')
        
        #Create Dataset or load it
        if not load_path:
            mention, left_side, right_side, label = pt.parse_dataset()

            dataset = TypingBERTDataSet(mention, left_side, right_side, label, id2label, label2id, vocab_len)
        else:
            with open(load_path, "rb") as filino:
                dataset = pickle.load(filino)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle, num_workers=10)
        
        # save dataset for future training
        if save_path:
            with open(save_path, "wb") as filino:
                pickle.dump(dataset, filino)

        if not train:
            return dataloader
        else:
            return dataloader, id2label, label2id, vocab_len