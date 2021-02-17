import torch
from transformers import BertModel
import torch.nn as nn


class Zeshel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = encoder.config.hidden_size
        self.scorelayer = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(self.hidden_dim, 1)
        )
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")

        # initialization of the layers is specific to bert-base-uncased
        self.scorelayer[1].weight.data.normal_(
            mean=0.0, std=self.encoder.config.initializer_range
        )
        self.scorelayer[1].bias.data.fill_(0.0)

    def forward(self, encoded_pairs, type_tokens, mention_masks, input_len):
        B, C, T = encoded_pairs.size()

        outputs = self.encoder(
            input_ids=encoded_pairs.view(-1, T).long(),
            token_type_ids=type_tokens.view(-1, T).long(),
        )[1]

        scores = self.scorelayer(outputs).unsqueeze(1).view(B, C)
        scores = scores.masked_fill_(input_len == 0, float("-inf"))
        # the target is [0]*B
        # if the input_len is 0, then the example is null
        loss = self.loss_fct(scores, torch.zeros(B).long().to(scores.device))

        max_scores, predictions = scores.max(dim=1)

        return {
            "loss": loss,
            "predictions": predictions,
            "max_scores": max_scores,
            "scores": scores,
        }
