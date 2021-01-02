from build.lib.typing_model import data
from typing_model.data.utils import DatasetParser
from typing_model.data.dataset import OnlyContextBERTDataset, TypingBERTDataSet, PaddedTypingBERTDataSet, ConcatenatedContextTypingBERTDataSet, OnlyMentionBERTDataset
from torch.utils.data import DataLoader
from typing_model.models.baseline import BaseBERTTyper, OnlyContextBERTTyper, TransformerBERTTyper, TransformerWHierarchicalLoss, ConcatenatedContextBERTTyper, TransformerWHierarchicalRegularization, OnlyMentionBERTTyper
from pytorch_lightning.callbacks import ModelCheckpoint

import torch

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

	def __init__(self) -> None:
		# declare the classes used in this experiment
		self.network_class = None
		self.dataset_class = None

	def setup(self, dataclass):
		# setup all class variables from dataclass and variables needed for instantiate the model
		raise NotImplementedError

	def perform_experiment(self):
		# perform the experiment
		raise NotImplementedError

	def instance_model(self):
		# instantiate the experiment's NN model
		raise NotImplementedError

class BaseTypingExperimentClass(BaseExperimentClass):

	def __init__(self, dataclass):
		self.train_data_path = dataclass.train_data_path
		self.eval_data_path = dataclass.eval_data_path
		self.test_data_path = dataclass.test_data_path
		
		self.early_stopping = dataclass.early_stopping
		self.early_stopping_patience = dataclass.early_stopping_patience
		self.early_stopping_metric = dataclass.early_stopping_metric
		self.early_stopping_mode = dataclass.early_stopping_mode
		self.epochs = dataclass.epochs

		self.load_train_dataset_path = dataclass.load_train_dataset_path
		self.load_eval_dataset_path = dataclass.load_eval_dataset_path
		self.load_test_dataset_path = dataclass.load_test_dataset_path
		
		self.save_train_dataset_path = dataclass.save_train_dataset_path
		self.save_eval_dataset_path = dataclass.save_eval_dataset_path
		self.save_test_dataset_path = dataclass.save_test_dataset_path

		self.checkpoint_monitor = dataclass.checkpoint_monitor
		self.checkpoint_folder_path = dataclass.checkpoint_folder_path
		self.checkpoint_name = dataclass.checkpoint_name
		self.checkpoint_mode = dataclass.checkpoint_mode
		self.save_auxiliary_variables = dataclass.save_auxiliary_variables
		self.aux_save_path = dataclass.aux_save_path
		self.auxiliary_variables_path = dataclass.auxiliary_variables_path

		self.weighted = dataclass.weighted
		self.weights_path = dataclass.weights_path

		self.max_mention_size = dataclass.max_mention_size
		self.max_context_size = dataclass.max_context_size

	def setup(self):
		self.dataloader_train, self.id2label, self.label2id, self.vocab_len = self.get_dataloader_from_dataset_path(self.train_data_path, 
																								shuffle=True, train = True,
																								load_path=self.load_train_dataset_path,
																								save_path=self.save_train_dataset_path)

		self.dataloader_val = self.get_dataloader_from_dataset_path(self.eval_data_path,
																id2label=self.id2label, label2id=self.label2id, vocab_len=self.vocab_len,
																load_path=self.load_eval_dataset_path,
																save_path=self.save_eval_dataset_path)

		if self.save_auxiliary_variables:
			with open(self.aux_save_path, "wb") as filino:
				pickle.dump((self.id2label, self.label2id, self.vocab_len), filino)


		if self.weighted:
			with open(self.weights_path, 'rb') as inp:
				weights = pickle.load(inp) 
			self.ordered_weights = torch.tensor([weights[self.id2label[i]] for i in range(len(self.id2label))])
		else:
			self.ordered_weights = None

		self.instance_model()

		self.declare_trainer_and_callbacks()

	def instance_model(self):
		self.bt = self.network_class(self.vocab_len, self.id2label, self.label2id, weights=self.ordered_weights, 
										max_mention_size = self.max_mention_size, max_context_size = self.max_context_size).cuda()
		
	def declare_trainer_and_callbacks(self):
		# declare callbacks
		callbacks = []

		if self.early_stopping:
			early_stop_callback = EarlyStopping(
			monitor=self.early_stopping_metric,
			min_delta=0.00,
			patience=self.early_stopping_patience,
			verbose=False,
			mode=self.early_stopping_mode
			)
			callbacks.append(early_stop_callback)
		
		checkpoint_callback = ModelCheckpoint(monitor=self.checkpoint_monitor,
												dirpath=self.checkpoint_folder_path,
												filename=self.checkpoint_name,
												mode=self.checkpoint_mode)
		callbacks.append(checkpoint_callback)
		self.trainer = Trainer(callbacks=callbacks)

	def perform_experiment(self):
		self.trainer.fit(self.bt, self.dataloader_train, self.dataloader_val)

	def get_dataloader_from_dataset_path(self, dataset_path, batch_size = 500, shuffle = False, train = False, load_path = None,
												save_path = None, id2label = None, label2id = None, vocab_len = None):
		
		pt = DatasetParser(dataset_path)

		if train and not load_path:
			id2label, label2id, vocab_len = pt.collect_global_config()
		elif load_path:
			with open(self.auxiliary_variables_path, 'rb') as filino:
				id2label, label2id, vocab_len = pickle.load(filino)
		elif not id2label or not label2id or not vocab_len:
			raise Exception('Please provide id2label_dict, label2id_dict and vocab len to generate val_loader or test_loader')

		#Create Dataloader or load it
		if not load_path:
			mention, left_side, right_side, label = pt.parse_dataset()

			dataset = self.dataset_class(mention, left_side, right_side, label, id2label, label2id, vocab_len, 
											self.max_mention_size, self.max_context_size)
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle, num_workers=10)
		else:
			with open(load_path, "rb") as filino:
				dataloader = pickle.load(filino)
			

		# save dataloader for future training
		if save_path:
			with open(save_path, "wb") as filino:
				pickle.dump(dataloader, filino)

		if not train:
			return dataloader
		else:
			return dataloader, id2label, label2id, vocab_len

class BertBaselineExperiment(BaseTypingExperimentClass):

	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = TypingBERTDataSet
		self.network_class = BaseBERTTyper

class BertTransformerExperiment(BaseTypingExperimentClass):

	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = PaddedTypingBERTDataSet
		self.network_class = TransformerBERTTyper

class ConcatenatedContextBERTTyperExperiment(BaseTypingExperimentClass):
	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = ConcatenatedContextTypingBERTDataSet
		self.network_class = ConcatenatedContextBERTTyper

class BertHierarchicalExperiment(BaseTypingExperimentClass):

	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = PaddedTypingBERTDataSet
		self.network_class = TransformerWHierarchicalLoss

		self.hierarchical_mode = dataclass.hierarchical_mode
		self.label_dependency_path = dataclass.label_dependency_path

	def instance_model(self):
		self.bt = self.network_class(self.vocab_len, self.id2label, self.label2id, weights=self.ordered_weights, 
											mode=self.hierarchical_mode, dependecy_file_path = self.label_dependency_path).cuda()

class BertHierarchicalRegularizedExperiment(BertHierarchicalExperiment):
	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = PaddedTypingBERTDataSet
		self.network_class = TransformerWHierarchicalRegularization

		self.hierarchical_mode = dataclass.hierarchical_mode
		self.label_dependency_path = dataclass.label_dependency_path

class BertOnlyMentionExperiment(BaseTypingExperimentClass):
	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = OnlyMentionBERTDataset
		self.network_class = OnlyMentionBERTTyper
		self.mention_max_length = dataclass.max_mention_size

	def get_dataloader_from_dataset_path(self, dataset_path, batch_size = 500, shuffle = False, train = False, load_path = None,
												save_path = None, id2label = None, label2id = None, vocab_len = None):
		
		pt = DatasetParser(dataset_path)

		if train and not load_path:
			id2label, label2id, vocab_len = pt.collect_global_config()
		elif load_path:
			with open(self.auxiliary_variables_path, 'rb') as filino:
				id2label, label2id, vocab_len = pickle.load(filino)
		elif not id2label or not label2id or not vocab_len:
			raise Exception('Please provide id2label_dict, label2id_dict and vocab len to generate val_loader or test_loader')

		#Create Dataloader or load it
		if not load_path:
			mention, left_side, right_side, label = pt.parse_dataset()

			dataset = self.dataset_class(mention, label, id2label, label2id, vocab_len, mention_max_tokens = self.mention_max_length)
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle, num_workers=10)
		else:
			with open(load_path, "rb") as filino:
				dataloader = pickle.load(filino)
			

		# save dataloader for future training
		if save_path:
			with open(save_path, "wb") as filino:
				pickle.dump(dataloader, filino)

		if not train:
			return dataloader
		else:
			return dataloader, id2label, label2id, vocab_len

	def instance_model(self):
		self.bt = self.network_class(self.vocab_len, self.id2label, self.label2id, weights=self.ordered_weights, 
											max_mention_size=self.mention_max_length).cuda()

class BertOnlyContextExperiment(BaseTypingExperimentClass):
	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = OnlyContextBERTDataset
		self.network_class = OnlyContextBERTTyper
		self.context_max_length = dataclass.max_context_size

	def get_dataloader_from_dataset_path(self, dataset_path, batch_size = 500, shuffle = False, train = False, load_path = None,
												save_path = None, id2label = None, label2id = None, vocab_len = None):
		
		pt = DatasetParser(dataset_path)

		if train and not load_path:
			id2label, label2id, vocab_len = pt.collect_global_config()
		elif load_path:
			with open(self.auxiliary_variables_path, 'rb') as filino:
				id2label, label2id, vocab_len = pickle.load(filino)
		elif not id2label or not label2id or not vocab_len:
			raise Exception('Please provide id2label_dict, label2id_dict and vocab len to generate val_loader or test_loader')

		#Create Dataloader or load it
		if not load_path:
			mentions, left_side, right_side, label = pt.parse_dataset()

			dataset = self.dataset_class(left_side, right_side, label, id2label, label2id, vocab_len, 
											context_max_tokens = self.context_max_length)
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle, num_workers=10)
		else:
			with open(load_path, "rb") as filino:
				dataloader = pickle.load(filino)
			

		# save dataloader for future training
		if save_path:
			with open(save_path, "wb") as filino:
				pickle.dump(dataloader, filino)

		if not train:
			return dataloader
		else:
			return dataloader, id2label, label2id, vocab_len

	def instance_model(self):
		self.bt = self.network_class(self.vocab_len, self.id2label, self.label2id, weights=self.ordered_weights, 
											max_context_size=self.context_max_length).cuda()
