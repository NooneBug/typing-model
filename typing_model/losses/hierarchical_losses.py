import torch
from random import sample

class HierarchicalLoss():

	def __init__(self, mode, id2label, label2id, label_dependency_file_path) :
		super().__init__()
		self.mode = mode
		self.id2label = id2label
		self.label2id = label2id
		self.label_dependency_file_path = label_dependency_file_path
		self.father_dict = {}
		self.NO_FATHER_VALUE = -1
		self.load_dependencies()

	def load_dependencies(self):
		# read the file which have node - father(node) couples and create a dict of node_index: father_node_index
		# if a node does not have a father, the dict will be node_index: self.NO_FATHER_VALUE
		with open(self.label_dependency_file_path, 'r') as inp:
			lines = [l.replace('\n', '').split('\t') for l in inp.readlines()]
			father_dict = {l[0]:l[1] for l in lines}
			for k, v in father_dict.items():
				if v in self.label2id:
					self.father_dict[self.label2id[k]] = self.label2id[v]
				else:
					 self.father_dict[self.label2id[k]] = self.NO_FATHER_VALUE

	# TODO: compute and log Hierarchy Violations
	def compute_loss(self, losses, logits, targets):
		penalty_tensor = []
		for ex_logits, ex_targets in zip(logits, targets):
		
			penalties = [1. for i in range(len(losses[0]))]

			positive_idxs = (ex_targets == 1).nonzero()

			for idx in positive_idxs:
				father = self.father_dict[idx.item()]
				if father != self.NO_FATHER_VALUE:
					father_logit = ex_logits[father].item()
					this_logit = ex_logits[idx].item()
					if this_logit > father_logit:
						if self.mode == 'absolute':
							penalties[father] = 1 + this_logit
						elif self.mode == 'relative':
							penalties[father] = 1 + (this_logit - father_logit)

			penalty_tensor.append(torch.tensor(penalties))
		
		return torch.mean(losses * torch.stack(penalty_tensor).cuda())

class HierarchicalRegularization():

	def __init__(self, mode, id2label, label2id, label_dependency_file_path) :
		super().__init__()
		self.mode = mode
		self.id2label = id2label
		self.label2id = label2id
		self.label_dependency_file_path = label_dependency_file_path
		self.father_dict = {}
		self.NO_FATHER_VALUE = -1
		self.n = 15
		self.lambda_value = 10
		self.load_dependencies()

	def load_dependencies(self):
		# read the file which have node - father(node) couples and create a dict of node_index: father_node_index
		# if a node does not have a father, the dict will be node_index: self.NO_FATHER_VALUE
		with open(self.label_dependency_file_path, 'r') as inp:
			lines = [l.replace('\n', '').split('\t') for l in inp.readlines()]
			father_dict = {l[0]:l[1] for l in lines}
			for k, v in father_dict.items():
				if v in self.label2id:
					self.father_dict[self.label2id[k]] = self.label2id[v]
				else:
					self.father_dict[self.label2id[k]] = self.NO_FATHER_VALUE

	def compute_loss(self, logits, targets):
		penalty_tensor = []
		
		for ex_logits, ex_targets in zip(logits, targets):
		
			penalties = [0. for i in range(self.n)]

			sampled_idxs = sample(range(len(ex_logits)), self.n)

			for i, idx in enumerate(sampled_idxs):
				ancestor = self.father_dict[idx]
				while ancestor != self.NO_FATHER_VALUE:
					ancestor_logit = ex_logits[ancestor].item()
					this_logit = ex_logits[idx].item()
					if this_logit > ancestor_logit:
						penalties[i] += this_logit - ancestor_logit
					ancestor = self.father_dict[ancestor]

			penalty_tensor.append(torch.tensor(penalties))
		
		return self.lambda_value * torch.mean(torch.stack(penalty_tensor).cuda())