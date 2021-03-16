from typing_model.runner.BERT_experimenters import ConcatenatedContextBERTTyperExperiment
from typing_model.models.joint_space_models import BaseJointSpaceModel, ExplicitClassifier, OneHotInjectedInferenceJointModel, PartialInitializedJointModel, RandomInitializedJointModel, TrainableOneHotModel
from typing_model.data.joint_spaces_datasets import RandomSingleLabelDataset

class JointSpaceBaseExperimenter(ConcatenatedContextBERTTyperExperiment):
	def __init__(self, dataclass):
		super().__init__(dataclass)
		
		self.joint_space_dim = dataclass.joint_space_dim
		self.trainable_threshold = dataclass.trainable_threshold
		self.random_threshold = dataclass.random_threshold
		self.threshold_value = dataclass.threshold_value
		self.inject_inference_in_model = dataclass.inject_inference_in_model
		self.regularization = dataclass.regularization

		self.dataset_class = RandomSingleLabelDataset
		self.network_class = BaseJointSpaceModel

	def instance_model(self):
		self.bt = self.network_class(self.vocab_len, self.id2label, self.label2id, 
									weights=self.ordered_weights,
										max_mention_size = self.max_mention_size, max_context_size = self.max_context_size,
										lr = self.learning_rate, bert_fine_tuning = self.bert_fine_tuning,
										loss_multiplier = self.loss_multiplier, positive_weight = self.positive_weight,
										joint_space_dim = self.joint_space_dim, trainable_threshold = self.trainable_threshold,
										random_threshold = self.random_threshold, threshold_value = self.threshold_value, 
										inject_inference_in_model=self.inject_inference_in_model,
										regularization=self.regularization ).cuda()

class ExplicitClassifierExperimenter(JointSpaceBaseExperimenter):
	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = RandomSingleLabelDataset
		self.network_class = ExplicitClassifier

class OneHotInjectedInferenceExperimenter(JointSpaceBaseExperimenter):
	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = RandomSingleLabelDataset
		self.network_class = OneHotInjectedInferenceJointModel

class TrainableOneHotExperimenter(JointSpaceBaseExperimenter):
	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = RandomSingleLabelDataset
		self.network_class = TrainableOneHotModel

class RandomInitializedJointExperimenter(JointSpaceBaseExperimenter):
	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = RandomSingleLabelDataset
		self.network_class = RandomInitializedJointModel

class PartialRandomInitializedJointExperimenter(JointSpaceBaseExperimenter):
	def __init__(self, dataclass):
		super().__init__(dataclass)
		self.dataset_class = RandomSingleLabelDataset
		self.network_class = PartialInitializedJointModel
		self.label_init_path = dataclass.label_init_path
	
	def instance_model(self):
		self.bt = self.network_class(self.vocab_len, self.id2label, self.label2id, 
									weights=self.ordered_weights,
										max_mention_size = self.max_mention_size, max_context_size = self.max_context_size,
										lr = self.learning_rate, bert_fine_tuning = self.bert_fine_tuning,
										loss_multiplier = self.loss_multiplier, positive_weight = self.positive_weight,
										joint_space_dim = self.joint_space_dim, trainable_threshold = self.trainable_threshold,
										random_threshold = self.random_threshold, threshold_value = self.threshold_value, 
										inject_inference_in_model=self.inject_inference_in_model, regularization = self.regularization,
										label_init_path=self.label_init_path ).cuda()