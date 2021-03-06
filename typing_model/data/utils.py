import json
# from stanfordcorenlp import StanfordCoreNLP
import ast
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing_model import data

class TreeMaker():
    def __init__(self):
        self.nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05')

    def parse(self, text):
        try:
            try:
                parsed = ""
                props = {'annotators': 'parse', 'outputFormat': 'json'}
                output = self.nlp.annotate(text, properties=props)
            except Exception:
                print("Except!!")
                return "(S)"
            # output = nlp.annotate(text, properties=props)
            # print(output)
            outputD = ast.literal_eval(output)
            senteces = outputD['sentences']
            if len(senteces) <= 1:
                root = senteces[0]['parse'].strip('\n')
                root = root.split(' ', 1)[1]
                root = root[1:len(root) - 1]
            else:
                s1 = senteces[0]['parse'].strip('\n')
                s1 = s1.split(' ', 1)[1]
                s1 = s1[1:len(s1) - 1]
                root = "(S" + s1
                for sentence in senteces[1:]:
                    s2 = sentence['parse'].strip('\n')
                    s2 = s2.split(' ', 1)[1]
                    s2 = s2[1:len(s2) - 1]
                    root = root + s2
                root = root + ")"

            return root.replace("\n", "")
        except Exception:
            print("Except")
            return "(S)"


class DatasetParser:

    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths

    def check_and_add_label_into_dict(self, labels):
        for label in labels:
            if len(self.label2id) == 0:
                self.label2id[label] = 0
            if label not in self.label2id:
                self.label2id[label] = max(list(self.label2id.values()))

    def get_id2label(self):
        return {v : k for k, v in self.label2id.items()}

    def collect_global_config(self):
        self.id2label = {}
        self.label2id = {}

        for dataset_path in self.dataset_paths.values():
            with open(dataset_path, 'r') as inp:
                lines = [json.loads(l) for l in inp.readlines()]

            ys = [k["y_str"] for k in lines]

            flat_list = set([item for sublist in ys for item in sublist])
            
            for label in flat_list:
                if label not in self.label2id:
                    new_id = len(self.id2label)
                    self.id2label[new_id] = label
                    self.label2id[label] = new_id

        return self.id2label, self.label2id, len(self.label2id)

    def parse_dataset(self, dataset_path = None, train_or_dev = None):
        '''
        if dataset_path is a single path the train_or_dev has to be None
        if dataset_path is a dictionary of path, passed here or in initialization,  
            then train_or_dev has to contain the key to extract the single dataset from the dict (suggested keys: train, dev, test)
        '''

        mentions = []
        left_context = []
        right_context = []
        labels = []

        if not dataset_path:
            dataset_path = self.dataset_paths

        if train_or_dev:
            with open(dataset_path[train_or_dev], 'r') as inp:
                lines = [json.loads(l) for l in inp.readlines()]
        else:
            with open(dataset_path, 'r') as inp:
                lines = [json.loads(l) for l in inp.readlines()]

        for l in lines:
            mentions.append((l['mention_span']))
            left_context.append((" ".join(l['left_context_token'])))
            right_context.append((" ".join(l['right_context_token'])))
            labels.append(l['y_str'])

        return mentions, left_context, right_context, labels

class GloveManager():
    def load_embedding_dict(self, embedding_path, embedding_size = 300):
        print("Loading word embeddings from {}...".format(embedding_path))
        default_embedding = np.zeros(embedding_size)
        embedding_dict = defaultdict(lambda: default_embedding)
        with open(embedding_path) as f:
            for line in tqdm(f.readlines()):
                splits = line.split()
                if len(splits) != embedding_size + 1:
                    continue
                assert len(splits) == embedding_size + 1
                word = splits[0]
                embedding = np.array([float(s) for s in splits[1:]])
                embedding_dict[word] = embedding
        print("Done loading word embeddings!")
        return embedding_dict