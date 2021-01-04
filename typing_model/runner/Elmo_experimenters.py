from dataclasses import dataclass
from typing_model.runner.base_experimenters import BaseTypingExperimentClass
from typing_model.models.ELMO_models import ElmoTyper
from typing_model.data.ELMo_datasets import ElmoConcatenatedContextDataset

class ElmoBaseExperiment(BaseTypingExperimentClass):

	def __init__(self, dataclass):
		super().__init__(dataclass)
		
		self.elmo_weight_file = dataclass.elmo_weight_file
		self.option_file = dataclass.option_file

		self.dataset_class = ElmoConcatenatedContextDataset
		self.network_class = ElmoTyper

	def instance_model(self):
		self.bt = self.network_class(self.vocab_len, self.id2label, self.label2id, class_weights=self.ordered_weights, 
										option_file = self.option_file, elmo_weights = self.elmo_weight_file)
	
	def instance_dataset(self):
		return self.dataset_class(self.mention, self.left_side, self.right_side, self.label, self.id2label, 
									self.label2id, self.vocab_len,
									max_mention_size = self.max_mention_size, max_context_size = self.max_context_size)