import sentence_transformers
from torch import nn
from sentence_transformers import SentenceTransformer


class BaseBERTTyper(nn.Module):

    def __init__(self):
        super().__init__()
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    def forward(self, mention, left, right):
        pass

