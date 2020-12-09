import json
from collections import defaultdict

class DatasetParser:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def check_and_add_label_into_dict(self, labels):
        for label in labels:
            if len(self.label2id) == 0:
                self.label2id[label] = 0
            if label not in self.label2id:
                self.label2id[label] = max(list(self.label2id.values()))

    def get_id2label(self):
        return {v : k for k, v in self.label2id.items()}

    def collect_global_config(self):
        with open(self.dataset_path, 'r') as inp:
            lines = [json.loads(l) for l in inp.readlines()]

        ys = [k["y_str"] for k in lines]

        flat_list = set([item for sublist in ys for item in sublist])

        self.id2label = {k : v for k, v in zip(range(0, len(flat_list)), flat_list)}
        self.label2id = {k : v for v, k in zip(range(0, len(flat_list)), flat_list)}

        return self.id2label, self.label2id, len(self.label2id)

    def parse_dataset(self, dataset_path = None):
        mentions = []
        left_context = []
        right_context = []
        labels = []

        if not dataset_path:
            dataset_path = self.dataset_path

        with open(dataset_path, 'r') as inp:
            lines = [json.loads(l) for l in inp.readlines()]

        for l in lines:
            mentions.append((l['mention_span']))
            left_context.append((" ".join(l['left_context_token'])))
            right_context.append((" ".join(l['right_context_token'])))
            labels.append(l['y_str'])


        return mentions, left_context, right_context, labels
