from typing_model.data.utils import DatasetParser
from typing_model.data.BERT_datasets import OnlyContextBERTDataset, TypingBERTDataSet, PaddedTypingBERTDataSet, ConcatenatedContextTypingBERTDataSet, OnlyMentionBERTDataset
from torch.utils.data import DataLoader
from typing_model.models.BERT_models import BaseBERTTyper, OnlyContextBERTTyper, TransformerWHierarchicalLoss, ConcatenatedContextBERTTyper, TransformerWHierarchicalRegularization, OnlyMentionBERTTyper
import pickle
from typing_model.runner.base_experimenters import BaseTypingExperimentClass


class BertBaselineExperiment(BaseTypingExperimentClass):

	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = TypingBERTDataSet
		self.network_class = BaseBERTTyper

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
