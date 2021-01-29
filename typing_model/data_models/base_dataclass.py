from dataclasses import dataclass
from torch.utils import data

@dataclass
class DataloaderDataclass:
    train_data_path : str = ''
    eval_data_path : str = ''
    test_data_path : str = ''

    save_train_dataset_path : str = None
    save_eval_dataset_path : str = None
    save_test_dataset_path : str = None
    
    load_train_dataset_path : str = None
    load_eval_dataset_path : str = None
    load_test_dataset_path : str = None

    save_auxiliary_variables : bool = False
    aux_save_path : str = None
    auxiliary_variables_path : str = None

    def __post_init__(self):
        self.save_auxiliary_variables = bool(self.save_auxiliary_variables)



@dataclass
class BaseDataclass(DataloaderDataclass):
    
    lr : float = 1e-3
    loss_multiplier : int = 1

    checkpoint_monitor : str = ''
    checkpoint_folder_path : str = ''
    checkpoint_name : str = ''
    checkpoint_mode : str = ''
    
    epochs : int = 1000
    min_epochs : int = 1
    early_stopping : bool = ''
    early_stopping_patience : int = 0
    early_stopping_metric : str = 'val_loss'
    early_stopping_mode : str = 'min'

    weighted : bool = False
    weights_path : str = None

    hierarchical_mode : str = None
    label_dependency_path : str = None

    train_batch_size : int = 500
    eval_batch_size : int = 500

    max_mention_size : int = None
    max_context_size : int = None

    experiment_name : str = None

    bert_fine_tuning : bool = False
    load_pretrained : bool = False
    pretrained_class_number : int = None
    state_dict_path : str = None
    fine_tuning : bool = False

    def __post_init__(self):
        self.early_stopping = bool(self.early_stopping)
        self.early_stopping_patience = int(self.early_stopping_patience)
        self.epochs = int(self.epochs)
        self.min_epochs = int(self.min_epochs)
        self.lr = float(self.lr)
        self.weighted = bool(self.weighted)
        if self.max_mention_size:
            self.max_mention_size = int(self.max_mention_size)
        if self.max_context_size:
            self.max_context_size = int(self.max_context_size)
        self.train_batch_size = int(self.train_batch_size)
        self.eval_batch_size = int(self.eval_batch_size)
        self.load_pretrained = bool(self.load_pretrained)
        self.fine_tuning = bool(self.fine_tuning)
        if self.pretrained_class_number:
            self.pretrained_class_number = int(self.pretrained_class_number)
        self.bert_fine_tuning = bool(self.bert_fine_tuning)
        self.loss_multiplier = float(self.loss_multiplier)

@dataclass
class ElmoDataclass(BaseDataclass):
    option_file : str = None
    elmo_weight_file : str = None

    def __post_init__(self):
        super().__post_init__()

        if not self.option_file:
            raise Exception('Please, provide an option file for ELMo model')
        if not self.elmo_weight_file:
            raise Exception('Please, provide a weight file for ELMo model')