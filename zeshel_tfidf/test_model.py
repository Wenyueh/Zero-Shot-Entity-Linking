import unittest
import torch
from model import Zeshel
from transformers import BertModel
import torch.nn as nn
import numpy as np
import random


class TestModel(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoded_pairs = torch.tensor(
            [
                [
                    [101, 2182, 2003, 1037, 3899, 102, 2008, 2003, 1037, 4937, 102],
                    [102, 2183, 2004, 1038, 3890, 102, 2009, 2004, 1038, 4938, 103],
                    [101, 2182, 2003, 1037, 3899, 102, 2008, 2003, 1037, 4937, 102],
                ],
                [
                    [101, 2182, 2003, 1037, 3899, 102, 2008, 2003, 1037, 4937, 102],
                    [102, 2183, 2004, 1038, 3890, 102, 2009, 2004, 1038, 4938, 103],
                    [101, 2182, 2003, 1037, 3899, 102, 2008, 2003, 1037, 4937, 102],
                ],
            ]
        ).to(self.device)
        self.type_tokens = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                ],
            ]
        ).to(self.device)
        self.mention_mark = torch.tensor(
            [
                [
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                ],
            ]
        ).to(self.device)
        self.input_len = torch.tensor([[11, 11, 11], [11, 11, 11]]).to(self.device)

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.model = Zeshel(self.encoder).to(self.device)

    def test_forward(self):
        self.model.eval()

        result = self.model(
            self.encoded_pairs, self.type_tokens, self.mention_mark, self.input_len
        )

        prediction = [0, 0]
        scores = [
            [0.5352581739425659, -0.11914516985416412, 0.5352581739425659],
            [0.5352581739425659, -0.11914516985416412, 0.5352581739425659],
        ]

        self.assertListEqual(result["predictions"].tolist(), prediction)
        self.assertListEqual(result["scores"].tolist(), scores)


if __name__ == "__main__":
    unittest.main()
