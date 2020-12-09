#!/usr/bin/env python

"""Tests for `typing_model` package."""

import unittest
from typing_model.data.dataset import *
from typing_model.data.parse_dataset import DatasetParser

class TestTyping_model(unittest.TestCase):
    """Tests for `typing_model` package."""


    def test_datasets(self):
        pt = DatasetParser("examples/toy_dataset.json")
        parsed = pt.parse_dataset()
        assert len(parsed[0]) == 2

        TypingBERTDataBase(*parsed)
