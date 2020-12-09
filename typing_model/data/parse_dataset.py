import json
from collections import defaultdict

class DatasetParser:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.label2id = defaultdict(int)

    def check_and_add_label_into_dict(self, labels):
        if len(self.label2id) == 0:
            self.label2id[label] = 0
        else:
            for label in labels:
                if label not in self.label2id:
                    self.label2id[label] = max(list(self.label2id.values()))

    def get_id2label(self):
        return { v : k for k, v in self.label2id.items()}

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
            self.check_and_add_label_into_dict(l['y_str'])

        return mentions, left_context, right_context, labels, self.label2id, self.get_id2label(), len(self.label2id)
