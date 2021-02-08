import json
import os
import math
from torch.utils.data import Dataset, DataLoader
import torch
from load_dataset import load_zeshel_data


class EntitySet(Dataset):
    def __init__(self, docs, max_len, tokenizer):
        self.docs = docs
        self.max_len = max_len
        self.ENT = "[unused2]"
        self.all_entities = list(docs.keys())
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.all_entities)

    def __getitem__(self, index):
        max_len = self.max_len - 2

        entity_id = self.all_entities[index]
        title = self.docs[entity_id]["title"]
        text = self.docs[entity_id]["text"]
        window = (
            self.tokenizer.tokenize(title) + [self.ENT] + self.tokenizer.tokenize(text)
        )[:max_len]

        encoded_window = self.tokenizer.encode_plus(
            window,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
        )

        encoded_id = torch.tensor(encoded_window["input_ids"]).long()
        encoded_msk = torch.tensor(encoded_window["attention_mask"]).long()

        print(encoded_id, encoded_msk)

        return encoded_id, encoded_msk


class MentionSet(Dataset):
    def __init__(self, mentions, docs, tokenizer, max_len):
        self.mentions = mentions
        self.docs = docs
        self.all_entities = list(self.docs.keys())
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.begin_boundary = "[unused0]"
        self.end_boundary = "[unused1]"

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, index):
        mention = self.mentions[index]
        mention_window = self.get_mention_window(mention)

        encoded_mention = self.tokenizer.encode_plus(
            mention_window,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
        )
        encoded_id = torch.tensor(encoded_mention["input_ids"]).long()
        encoded_attn = torch.tensor(encoded_mention["attention_mask"]).long()

        document_label = mention["label_document_id"]
        label = torch.tensor(self.all_entities.index(document_label)).long()

        return encoded_id, encoded_attn, label

    def get_mention_window(self, mention):
        tokens = self.docs[mention["context_document_id"]]["text"].split()

        max_len_mention = self.max_len - 2

        # compute the mention tokens
        mention_tokens = (
            [self.begin_boundary]
            + self.tokenizer.tokenize(mention["text"])
            + [self.end_boundary]
        )

        if len(mention_tokens) >= max_len_mention:
            return mention_tokens, 0, len(mention_tokens) - 1

        mention_length = len(mention_tokens)

        # compute the prefix tokens and suffix tokens
        context_length = math.ceil((max_len_mention - mention_length) / 2)

        prefix = tokens[
            max(mention["start_index"] - max_len_mention, 0) : mention["start_index"]
        ]
        suffix = tokens[
            mention["end_index"] + 1 : mention["end_index"] + 1 + max_len_mention
        ]
        prefix = self.tokenizer.tokenize(" ".join(prefix))
        suffix = self.tokenizer.tokenize(" ".join(suffix))

        if len(suffix) <= context_length:
            prefix = prefix[-(max_len_mention - mention_length - len(suffix)) :]
        else:
            prefix = prefix[-context_length:]

        window = prefix + mention_tokens + suffix
        window = window[:max_len_mention]

        return window


# the inputs are outputs of load_data function
class Data:
    def __init__(
        self,
        train_docs,
        val_docs,
        test_docs,
        debug_docs,
        train_mentions,
        val_mentions,
        test_mentions,
        debug_mentions,
        tokenizer,
        max_len,
    ):
        self.train_docs = train_docs
        self.val_docs = val_docs
        self.test_docs = test_docs
        self.debug_docs = debug_docs
        self.train_mentions = train_mentions
        self.val_mentions = val_mentions
        self.test_mentions = test_mentions
        self.debug_mentions = debug_mentions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def get_loaders(self, batch_size):
        train_entities = EntitySet(self.train_docs, self.max_len, self.tokenizer)
        val_entities = EntitySet(self.val_docs, self.max_len, self.tokenizer)
        test_entities = EntitySet(self.test_docs, self.max_len, self.tokenizer)
        debug_entities = EntitySet(self.debug_docs, self.max_len, self.tokenizer)

        train_mentions = MentionSet(
            self.train_mentions, self.train_docs, self.tokenizer, self.max_len
        )
        val_mentions = MentionSet(
            self.val_mentions, self.val_docs, self.tokenizer, self.max_len
        )
        test_mentions = MentionSet(
            self.test_mentions, self.test_docs, self.tokenizer, self.max_len
        )
        debug_mentions = MentionSet(
            self.debug_mentions, self.debug_docs, self.tokenizer, self.max_len
        )

        train_en_loader = DataLoader(train_entities, batch_size, shuffle=False)
        val_en_loader = DataLoader(val_entities, batch_size, shuffle=False)
        test_en_loader = DataLoader(test_entities, batch_size, shuffle=False)
        debug_en_loader = DataLoader(debug_entities, batch_size, shuffle=False)
        train_me_loader = DataLoader(train_mentions, batch_size, shuffle=False)
        val_me_loader = DataLoader(val_mentions, batch_size, shuffle=False)
        test_me_loader = DataLoader(test_mentions, batch_size, shuffle=False)
        debug_me_loader = DataLoader(debug_mentions, batch_size, shuffle=False)

        return (
            train_en_loader,
            val_en_loader,
            test_en_loader,
            debug_en_loader,
            train_me_loader,
            val_me_loader,
            test_me_loader,
            debug_me_loader,
        )
