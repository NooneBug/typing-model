from typing_model.data.base_dataset import TypingDataSet 
from transformers import BertTokenizer
import gc
import torch
import time

import numpy as np

class TypingBERTDataSet(TypingDataSet):

    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size):
        super().__init__(mentions, left_side, right_side, label, id2label, label2id, vocab_size)

class PaddedTypingBERTDataSet(TypingDataSet):

    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size):
        # TO DO: add hyperparameter to BERT encodings (max_length)
        super().__init__(mentions, left_side, right_side, label, id2label, label2id, vocab_size)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        print('encoding dataset...')

        t = time.time()
        self.mentions = [torch.tensor(t) for t in tokenizer(mentions,
                                                    padding='max_length',
                                                    max_length = 25,
                                                    truncation=True,
                                                    return_tensors="pt")['input_ids'].tolist()]

        print('mentions encoded in {} seconds'.format(round(time.time() - t, 2)))

        t = time.time()
        self.left_side = [torch.tensor(t) for t in tokenizer(left_side,
                                                    padding='max_length',
                                                    max_length = 25,
                                                    truncation=True,
                                                    return_tensors="pt")['input_ids'].tolist()]

        print('left_contexts encoded in {} seconds'.format(round(time.time() - t, 2)))


        t = time.time()
        self.right_side = [torch.tensor(t) for t in tokenizer(right_side,
                                                                padding='max_length',
                                                                max_length = 25,
                                                                truncation=True,
                                                                return_tensors="pt")['input_ids'].tolist()]
        print('right_contexts encoded in {} seconds'.format(round(time.time() - t, 2)))

        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

class OnlyMentionBERTDataset(TypingDataSet):
    def __init__(self, mentions, label, id2label, label2id, vocab_size, 
                    mention_max_tokens):

        self.label = label
        self.vocab_size = vocab_size

        self.id2label = id2label
        self.label2id = label2id
        self.label_id = [[self.label2id[v] for v in k] for k in self.label]
        self.mention_max_tokens = mention_max_tokens

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        print('encoding dataset...')


        t = time.time()
        self.mentions = [torch.tensor(t) for t in tokenizer(mentions,
                                                    padding='max_length',
                                                    max_length = self.mention_max_tokens,
                                                    truncation=True,
                                                    return_tensors="pt")['input_ids'].tolist()]

        print('mentions encoded in {} seconds'.format(round(time.time() - t, 2)))

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        labels_id = self.label_id[idx]
        one_hot = torch.zeros(self.vocab_size)
        one_hot[labels_id] = 1

        return self.mentions[idx], one_hot

class OnlyContextBERTDataset(TypingDataSet):
    def __init__(self, left_side, right_side, label, id2label, label2id, vocab_size, 
                    context_max_tokens):

        self.label = label
        self.vocab_size = vocab_size

        self.id2label = id2label
        self.label2id = label2id
        self.label_id = [[self.label2id[v] for v in k] for k in self.label]
        self.context_max_tokens = context_max_tokens

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        print('encoding dataset...')

        t = time.time()

        extracted_left_side = [' '.join(l.split(' ')[- int(np.floor(self.context_max_tokens/2)):]) for l in left_side]
        extracted_right_side = [' '.join(r.split(' ')[: int(np.floor(self.context_max_tokens/2))]) for r in right_side]

        contexts = [l + ' ' + r for l, r in zip(extracted_left_side, extracted_right_side)]
        self.contexts = [torch.tensor(t) for t in tokenizer(contexts,
                                                    padding='max_length',
                                                    max_length = self.context_max_tokens,
                                                    truncation=True,
                                                    return_tensors="pt")['input_ids'].tolist()]

        print('contexts encoded in {} seconds'.format(round(time.time() - t, 2)))

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        labels_id = self.label_id[idx]
        one_hot = torch.zeros(self.vocab_size)
        one_hot[labels_id] = 1

        return self.contexts[idx], one_hot


class ConcatenatedContextTypingBERTDataSet(TypingDataSet):

    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size, 
                    mention_max_tokens, context_max_tokens):

        super().__init__(mentions, left_side, right_side, label, id2label, label2id, vocab_size)
        self.label = label
        self.vocab_size = vocab_size

        self.id2label = id2label
        self.label2id = label2id
        self.label_id = [[self.get_label_from_id(v) for v in k] for k in self.label]
        self.print_out_of_train_labels_number()

        self.mention_max_tokens = mention_max_tokens
        self.context_max_tokens = context_max_tokens

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        print('encoding dataset...')


        t = time.time()
        self.mentions = [torch.tensor(t) for t in tokenizer(mentions,
                                                    padding='max_length',
                                                    max_length = self.mention_max_tokens,
                                                    truncation=True,
                                                    return_tensors="pt")['input_ids'].tolist()]

        print('mentions encoded in {} seconds'.format(round(time.time() - t, 2)))


        t = time.time()
        extracted_left_side = [' '.join(l.split(' ')[- int(np.floor(self.context_max_tokens/2)):]) for l in left_side]
        extracted_right_side = [' '.join(r.split(' ')[: int(np.floor(self.context_max_tokens/2))]) for r in right_side]

        contexts = [l + ' ' + r for l, r in zip(extracted_left_side, extracted_right_side)]
        self.contexts = [torch.tensor(t) for t in tokenizer(contexts,
                                                    padding='max_length',
                                                    max_length = self.context_max_tokens,
                                                    truncation=True,
                                                    return_tensors="pt")['input_ids'].tolist()]

        print('contexts encoded in {} seconds'.format(round(time.time() - t, 2)))
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, idx):
        labels_id = self.label_id[idx]
        one_hot = torch.zeros(self.vocab_size)
        one_hot[labels_id] = 1

        return self.mentions[idx], self.contexts[idx], one_hot


