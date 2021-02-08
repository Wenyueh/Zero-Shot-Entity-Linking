#!/usr/bin/env ipython

import torch
import torch.nn as nn


class DualEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__
        self.mention_encoder = encoder
        self.entity_encoder = copy.deepcopy(self.mention_encoder)

    def forward(self, mention, entity):
        encoded_mention = self.mention_encoder(
            input_ids=mention[0], attention_mask=mention[1]
        )
        encoded_entity = self.entity_encoder(
            input_ids=entity[0], attention_mask=entity[1]
        )
        mention_cls = encoded_mention[0][:, 0, :]
        entity_cls = encoded_entity[0][:, 0, :]

        logits = torch.mm(mention_cls, entity_cls.transpose(1, 0))
