import unittest
import torch
from model import Zeshel
from transformers import BertModel
import torch.nn as nn
import numpy as np
import random
import os

from transformers import AdamW


def count_parameters(model):
    number_parameters = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            number_parameters += torch.numel(p)

    return number_parameters


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

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.model2 = Zeshel(self.encoder).to(self.device)
        # self.model_orig = Zeshel_orig(self.encoder, 0, self.device, False, True).to(
        # self.device
        # )

    def test_forward(self):

        self.model.eval()
        self.model2.eval()

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        result = self.model(
            self.encoded_pairs, self.type_tokens, self.mention_mark, self.input_len
        )

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        result2 = self.model2(
            self.encoded_pairs, self.type_tokens, self.mention_mark, self.input_len
        )

        prediction = [0, 0]
        scores = [
            [0.5352581739425659, -0.11914516985416412, 0.5352581739425659],
            [0.5352581739425659, -0.11914516985416412, 0.5352581739425659],
        ]

        self.assertListEqual(result["predictions"].tolist(), prediction)
        self.assertListEqual(result["scores"].tolist(), scores)

        self.assertListEqual(result2["predictions"].tolist(), prediction)
        self.assertListEqual(result2["scores"].tolist(), scores)

    def test_initialization(self):
        for n, p in self.model.named_parameters():
            if "scorelayer" in n and "weight" in n:
                weight = p
            if "scorelayer" in n and "bias" in n:
                bias = p

        for n, p in self.model2.named_parameters():
            if "scorelayer" in n and "weight" in n:
                weight2 = p
            if "scorelayer" in n and "bias" in n:
                bias2 = p

        self.assertListEqual(
            weight.tolist(), weight2.tolist(), "Model weights are different"
        )
        self.assertListEqual(bias.tolist(), bias2.tolist(), "Model bias are different")

    def test_loss(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        result = self.model(
            self.encoded_pairs, self.type_tokens, self.mention_mark, self.input_len
        )

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        result2 = self.model2(
            self.encoded_pairs, self.type_tokens, self.mention_mark, self.input_len
        )

        loss = result["loss"]
        loss2 = result2["loss"]
        self.assertEqual(loss, loss2, "Loss are computed to be different")

    def test_grad(self):
        self.model.train()
        self.model2.train()

        self.model.zero_grad()
        self.model2.zero_grad()

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        result = self.model(
            self.encoded_pairs, self.type_tokens, self.mention_mark, self.input_len
        )

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        result2 = self.model2(
            self.encoded_pairs, self.type_tokens, self.mention_mark, self.input_len
        )

        loss = result["loss"]
        loss2 = result2["loss"]

        loss.backward()
        loss2.backward()

        for n, p in self.model.named_parameters():
            if "scorelayer" in n and "weight" in n:
                grad = p.grad.data

        for n, p in self.model2.named_parameters():
            if "scorelayer" in n and "weight" in n:
                grad2 = p.grad.data

        self.assertListEqual(
            grad.tolist(), grad2.tolist(), "Gradients are computed to be different"
        )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    unittest.main()
