#!/usr/bin/env ipython
import unittest
from main import evaluate
import torch


class TestEval(unittest.TestCase):
    def setUp(self):
        self.predictions = [
            torch.tensor([0, 1, 0, 1, 0, 1]),
            torch.tensor([1, 0, 1, 1, 1]),
            torch.tensor([1, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
            torch.tensor([0, 0]),
        ]
        self.accuracy = torch.tensor(0.6)

    def test_evaluate(self):
        result = evaluate(self.predictions)
        self.assertEqual(result, self.accuracy, "Macro evaluation wrong")


if __name__ == "__main__":
    unittest.main()
