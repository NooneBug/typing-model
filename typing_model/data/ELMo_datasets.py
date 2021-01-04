from typing_model.data.base_dataset import TypingDataSet 
from allennlp.modules.elmo import batch_to_ids
import torch 
import time

class ElmoDataset(TypingDataSet):
    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size):
        super().__init__(mentions, left_side, right_side, label, id2label, label2id, vocab_size)

        self.mentions = batch_to_ids([m.split(' ') for m in mentions])
        self.left_side = batch_to_ids(l.split(' ') for l in left_side)
        self.right_side = batch_to_ids(r.split(' ') for r in right_side )
        self.label = label


class ElmoConcatenatedContextDataset(TypingDataSet):
    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size, max_mention_size, max_context_size):
        super().__init__(mentions, left_side, right_side, label, id2label, label2id, vocab_size)
        
        print('Getting dataset ...')
        
        t = time.time()
        if max_mention_size:
            mentions = [m[:max_mention_size] for m in mentions]
        if max_context_size:
            context = [l[: - int(max_context_size/2)] + ' ' + r[: int(max_context_size/2)] for l, r in zip(left_side, right_side)]
        else:
            context = [l + ' ' + r for l, r in zip(left_side, right_side)]
        self.mentions = batch_to_ids([m.split(' ') for m in mentions])
        print('mentions tokenized in {} seconds'.format(round(time.time() - t ,2)))
        t = time.time()
        self.contexts = batch_to_ids(c.split(' ') for c in context)
        print('contexts tokenized in {} seconds'.format(round(time.time() - t ,2)))

        self.label = label

    def __getitem__(self, idx):
        labels_id = self.label_id[idx]
        one_hot = torch.zeros(self.vocab_size)
        one_hot[labels_id] = 1

        return self.mentions[idx], self.contexts[idx], one_hot