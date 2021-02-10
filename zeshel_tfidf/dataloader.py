#!/usr/bin/env ipython

import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
import math
from transformers import BertTokenizer


def get_loaders(
    data_path,
    batch,
    max_candidates,
    max_len,
    tokenizer,
    num_worker,
    indicate_mention_boundary,
):
    (
        documents,
        train_sample,
        heldout_train_seen_sample,
        heldout_train_unseen_sample,
        val_sample,
        test_sample,
    ) = load_zeshel_data(data_path)

    train_dataset = ZeshelDataset(
        documents,
        train_sample,
        max_candidates,
        max_len,
        tokenizer,
        True,
        indicate_mention_boundary,
    )
    heldout_train_seen_dataset = ZeshelDataset(
        documents,
        heldout_train_seen_sample,
        max_candidates,
        max_len,
        tokenizer,
        False,
        indicate_mention_boundary,
    )
    heldout_train_unseen_dataset = ZeshelDataset(
        documents,
        heldout_train_unseen_sample,
        max_candidates,
        max_len,
        tokenizer,
        False,
        indicate_mention_boundary,
    )
    val_dataset = ZeshelDataset(
        documents,
        val_sample,
        max_candidates,
        max_len,
        tokenizer,
        False,
        indicate_mention_boundary,
    )
    test_dataset = ZeshelDataset(
        documents,
        test_sample,
        max_candidates,
        max_len,
        tokenizer,
        False,
        indicate_mention_boundary,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch, num_workers=num_worker, shuffle=True
    )
    heldout_train_seen_loader = DataLoader(
        heldout_train_seen_dataset,
        batch_size=batch,
        num_workers=num_worker,
        shuffle=False,
    )
    heldout_train_unseen_loader = DataLoader(
        heldout_train_unseen_dataset,
        batch_size=batch,
        num_workers=num_worker,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch, num_workers=num_worker, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch, num_workers=num_worker, shuffle=False
    )

    return (
        train_loader,
        heldout_train_seen_loader,
        heldout_train_unseen_loader,
        val_loader,
        test_loader,
    )


class ZeshelDataset(Dataset):
    def __init__(
        self,
        documents,
        mention_candidate_pairs,
        max_candidates,
        max_len,
        tokenizer,
        is_training,
        indicate_mention_boundary,
    ):
        self.documents = documents
        self.max_len = max_len
        self.max_len_mention = (self.max_len - 3) // 2  # [CLS], [SEP], [SEP]
        self.max_len_candidate = self.max_len - 3 - self.max_len_mention
        self.begin_boundary = "[unused0]"
        self.end_boundary = "[unused1]"
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.max_candidates = max_candidates
        self.indicate_mention_boundary = indicate_mention_boundary
        self.samples = self.cull_samples(mention_candidate_pairs)

    def __len__(self):
        return len(self.mention_candidate_pairs)

    def __getitem__(self, index):
        mention, candidates = self.samples[index]
        window, start_idx, end_idx = self.get_mention_window(mention)
        start_idx += 1
        end_idx += 1

        candidate_ids = self.prepare_candidates(mention, candidates)

        self.encoded_pairs = torch.zeros(self.max_candidates, self.max_len)
        self.type_tokens = torch.zeros(self.max_candidates, self.max_len)
        self.mention_masks = torch.zeros(self.max_candidates, self.max_len)
        self.input_len = torch.zeros(self.max_candidates)

        for i, candidate_id in enumerate(candidate_ids):
            candidate_prefix = (
                self.tokenizer.tokenize(self.documents[candidate_id]["text"])
            )[: self.max_len_candidate]
            encoded = self.tokenizer.encode_plus(
                window,
                candidate_prefix,
                pad_to_max_length=self.max_len,
                truncation=True,
            )
            self.encoded_pairs[i] = torch.tensor(encoded["input_ids"])
            self.type_tokens[i] = torch.tensor(encoded["token_type_ids"])
            self.mention_masks[i][start_idx : end_idx + 1] = 1
            self.input_len[i] = 3 + len(window) + len(candidate_prefix)

        return self.encoded_pairs, self.type_tokens, self.mention_masks, self.input_len

    def cull_samples(self, mention_candidate_pairs):
        if self.is_training:
            return mention_candidate_pairs
        else:
            return [
                mc
                for mc in mention_candidate_pairs
                if mc[0]["label_document_id"]
                in mc[1]["tfidf_candidates"][: self.max_candidates]
            ]

    def prepare_candidates(self, mention, candidates):
        c = []
        if self.is_training:
            if (
                not mention["label_document_id"]
                in candidates["tfidf_candidates"][: self.max_candidates]
            ):
                c.append(mention["label_document_id"])
        c = c + candidates["tfidf_candidates"][: self.max_candidates]

        assert mention["label_document_id"] in c

        return c[: self.max_candidates]

    def get_mention_window(self, mention):
        tokens = self.documents[mention["context_document_id"]]["text"].split()

        # compute the mention tokens
        if self.indicate_mention_boundary:
            mention_tokens = (
                [self.begin_boundary]
                + self.tokenizer.tokenize(mention["text"])
                + [self.end_boundary]
            )
        else:
            mention_tokens = self.tokenizer.tokenize(mention["text"])

        prefix = tokens[
            max(mention["start_index"] - self.max_len_mention, 0) : mention[
                "start_index"
            ]
        ]
        suffix = tokens[
            mention["end_index"] + 1 : mention["end_index"] + 1 + self.max_len_mention
        ]
        prefix = self.tokenizer.tokenize(" ".join(prefix))
        suffix = self.tokenizer.tokenize(" ".join(suffix))

        return get_window(mention_tokens, prefix, suffix, self.max_len_mention)


def get_window(mention_tokens, prefix, suffix, max_len_mention):
    if len(mention_tokens) >= max_len_mention:
        return mention_tokens[:max_len_mention], 0, max_len_mention - 1

    mention_length = len(mention_tokens)

    # compute the prefix tokens and suffix tokens
    context_length = math.ceil((max_len_mention - mention_length) / 2)

    if len(suffix) <= context_length:
        prefix = prefix[-(max_len_mention - mention_length - len(suffix)) :]
    else:
        prefix = prefix[:context_length]

    window = prefix + mention_tokens + suffix
    window = window[:max_len_mention]

    start = len(prefix)
    end = len(prefix) + mention_length - 1

    return window, start, end


def load_zeshel_data(data_path):
    def load_documents(data_path):
        documents = {}
        for fname in os.listdir(data_path + "/documents"):
            with open(data_path + "/documents/" + fname + ".json", "r") as f:
                for line in f:
                    one_document = json.loads(line)
                    documents[one_document["document_id"]] = one_document
        return documents

    def load_mention_candidate_pairs(part):
        mentions = []
        with open(data_path + "/mentions/" + part, "r") as f:
            for line in f:
                one_mention = json.loads(line)
                mentions.append(one_mention)

        candidates = []
        with open(data_path + "/tfidf_candidates/" + part, "r") as f:
            for i, line in enumerate(f):
                one_candidate = json.loads(line)
                candidates.append(one_candidate)
                assert one_candidate["mention_id"] == mentions[i]["mention_id"]
        return zip(mentions, candidates)

    documents = load_documents(data_path)
    sample_train = load_mention_candidate_pairs("train")
    sample_heldout_train_seen = load_mention_candidate_pairs("heldout_train_seen")
    sample_heldout_train_unseen = load_mention_candidate_pairs("heldout_train_unseen")
    sample_val = load_mention_candidate_pairs("val")
    sample_test = load_mention_candidate_pairs("test")

    return (
        documents,
        sample_train,
        sample_heldout_train_seen,
        sample_heldout_train_unseen,
        sample_val,
        sample_test,
    )
