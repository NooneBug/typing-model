from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import gc
import torch
import time

class TypingDataSet(Dataset):

    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size):
        self.mentions = mentions
        self.left_side = left_side
        self.right_side = right_side
        self.label = label
        self.vocab_size = vocab_size

        self.id2label = id2label
        self.label2id = label2id
        self.label_id = [[self.label2id[v] for v in k] for k in self.label]

    def __len__(self):
        return len(self.left_side)

    def __getitem__(self, idx):
        labels_id = self.label_id[idx]
        one_hot = torch.zeros(self.vocab_size)
        one_hot[labels_id] = 1

        return self.mentions[idx], self.left_side[idx], self.right_side[idx], one_hot

    def to_cuda(self):
        self.mentions = self.mentions.cuda()
        self.left_side = self.left_side.cuda()
        self.right_side = self.right_side.cuda()



class TypingBERTDataSet(TypingDataSet):

    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size):
        super().__init__(mentions, left_side, right_side, label, id2label, label2id, vocab_size)
        model = SentenceTransformer('bert-base-uncased')

        self.mentions = model.encode(mentions, show_progress_bar=True, batch_size=100)
        self.left_side = model.encode(left_side, show_progress_bar=True, batch_size=100)
        self.right_side = model.encode(right_side, show_progress_bar=True, batch_size=100)

        del model
        torch.cuda.empty_cache()
        gc.collect()

class PaddedTypingBERTDataSet(TypingDataSet):

    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size):
        # TO DO: add hyperparameter to BERT encodings (max_length)
        super().__init__(mentions, left_side, right_side, label, id2label, label2id, vocab_size)
        # print('torch.cuda.current_device(): {}'.format(torch.cuda.current_device()))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = BertModel.from_pretrained("bert-base-uncased")

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
        
        
        # del model
        del tokenizer
        # del outs
        # self.to_cuda()
        torch.cuda.empty_cache()
        gc.collect()


class SimplerTypingBERTDataSet(Dataset):

    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size):
        
        self.label = label
        self.vocab_size = vocab_size

        self.id2label = id2label
        self.label2id = label2id
        self.label_id = [[self.label2id[v] for v in k] for k in self.label]

        # TO DO: add hyperparameter to BERT encodings (max_length)
        # print('torch.cuda.current_device(): {}'.format(torch.cuda.current_device()))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = BertModel.from_pretrained("bert-base-uncased")

        print('encoding dataset...')


        t = time.time()
        self.mentions = [torch.tensor(t) for t in tokenizer(mentions,
                                                    padding='max_length',
                                                    max_length = 25,
                                                    truncation=True, 
                                                    return_tensors="pt")['input_ids'].tolist()]

        print('mentions encoded in {} seconds'.format(round(time.time() - t, 2)))

        
        t = time.time()
        contexts = [l + ' ' + r for l, r in zip(left_side, right_side)]
        self.contexts = [torch.tensor(t) for t in tokenizer(contexts,
                                                    padding='max_length',
                                                    max_length = 50,
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
