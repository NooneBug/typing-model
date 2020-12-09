#!/usr/bin/env python

"""Tests for `typing_model` package."""

import unittest
from typing_model.data.dataset import *
from typing_model.data.parse_dataset import DatasetParser

class TestTyping_model(unittest.TestCase):
    """Tests for `typing_model` package."""

    def test_datasets(self):
        pt = DatasetParser("examples/toy_dataset.json")
        id2label, label2id, vocab_len = pt.collect_global_config()
        mention_train, left_side_train, right_side_train, label_train = pt.parse_dataset()
        assert len(mention_train) == 2

