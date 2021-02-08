#!/usr/bin/env ipython

import torch


def get_all_entity_embeddings(en_loader, model):
    encoder = model.entity_encoder
    entity_embeddings = []
    for i, entity in enumerate(en_loader):
        embedding = encoder(input_ids=entity[0], attention_mask=entity[1])[0][:, 0, :]
        entity_embeddings.append(embedding)
    entity_embeddings = torch.cat(embedding, dim=0)
    return entity_embeddings


def get_hard_negative(men_loader, entity_embeddings, model):
    encoder = model.mention_encoder
    for i, mention in enumerate(men_loader):
        mention_embedding = encoder(input_ids=mention[0], attention_mask=mention[1])[0][
            :, 0, :
        ]
        mention_embedding = torch.cat(
            [mention_embedding.unsqueeze(0)] * entity_embeddings.size(0), dim=0
        )
        logits = torch.bmm(mention_embedding, entity_embeddings)
