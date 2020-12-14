from dataclasses import dataclass

@dataclass
class BaseDataclass:
    train_data_path : str
    eval_data_path : str
    test_data_path : str

    # early_stopping : bool
    early_stopping_patience : int
    epochs : int

@dataclass
class BertBaselineDataclass(BaseDataclass):
    train_data_path : str 
    eval_data_path : str 
    test_data_path : str 
    early_stopping : bool
    early_stopping_patience : int
    epochs : int

    def __post_init__(self):
        self.early_stopping = bool(self.early_stopping)
        self.early_stopping_patience = int(self.early_stopping_patience)
        self.epochs = int(self.epochs)

    # early_stopping = bool(early_stopping)
    # early_stopping_patience = int(early_stopping_patience)
    # epochs = int(epochs)