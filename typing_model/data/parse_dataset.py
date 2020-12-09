import json

class DatasetParser:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

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
            left_context.append((" ".join(l['left_context_tokens'])))
            right_context.append((" ".join(l['right_context_tokens'])))
            labels.append(l['y_str'])

        return mentions, left_context, right_context, labels
