#!/usr/bin/env ipython
import os
import json


def load_entities(data_path, domains):
    documents = {}
    for fname in domains:
        with open(
            os.path.join(os.path.join(data_path, "documents"), fname + ".json"), "r"
        ) as f:
            for line in f:
                fields = json.loads(line)
                documents[fields["document_id"]] = fields
    return documents


def load_mentions(data_path, part):
    mentions = []
    domains = set()
    with open(os.path.join(data_path, "mentions/{}.json".format(part)), "r") as f:
        for i, line in enumerate(f):
            mention = json.loads(line)
            mentions.append(mention)
            domains.add(mention["corpus"])
    return mentions, domains


def load_zeshel_data(data_path):
    sample_train, train_domains = load_mentions(data_path, "train")
    sample_test, test_domains = load_mentions(data_path, "test")
    sample_val, val_domains = load_mentions(data_path, "val")
    sample_debug, debug_domains = load_mentions(data_path, "debug")
    sample_heldout_train_seen, heldout_train_seen_domains = load_mentions(
        data_path, "heldout_train_seen"
    )
    sample_heldout_train_unseen, heldout_train_unseen_domains = load_mentions(
        data_path, "heldout_train_unseen"
    )

    train_entities = load_entities(data_path, train_domains)
    test_entities = load_entities(data_path, test_domains)
    val_entities = load_entities(data_path, val_domains)
    debug_entities = load_entities(data_path, debug_domains)
    heldout_train_seen_entities = load_entities(data_path, heldout_train_seen_domains)
    heldout_train_unseen_entities = load_entities(
        data_path, heldout_train_unseen_domains
    )

    return (
        sample_train,
        sample_heldout_train_seen,
        sample_heldout_train_unseen,
        sample_val,
        sample_debug,
        sample_test,
        train_entities,
        heldout_train_seen_entities,
        heldout_train_unseen_entities,
        val_entities,
        debug_entities,
        test_entities,
    )
