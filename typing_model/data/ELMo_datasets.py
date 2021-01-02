from typing_model.data.base_dataset import TypingDataSet 


class ElmoDataset(TypingDataSet):
    def __init__(self, mentions, left_side, right_side, label, id2label, label2id, vocab_size):
        super().__init__(mentions, left_side, right_side, label, id2label, label2id, vocab_size)

        