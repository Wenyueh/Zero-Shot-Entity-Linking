#!/usr/bin/env ipython
import argparse
import torch
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
    get_constant_schedule,
)
import torch.nn as nn
from model import Zeshel
from dataloader import get_loaders, load_zeshel_data

import random
import numpy as np


def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def construct_optimizer(args, model, num_training_examples):
    no_weight_decay = ["LayerNorm", "bias"]
    for n, p in model.named_parameters():
        pass


def construct_optimizer_simple(args, model, num_training_examples):
    pass


def main():
    set_seed()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder = BertModel.from_pretrained("bert-base-uncased")

    data = load_zeshel_data(args.data_path)

    num_train_examples = len(data[1])

    (
        train_loader,
        heldout_train_seen_loader,
        heldout_train_unseen_loader,
        val_loader,
        test_loader,
    ) = get_loaders(
        args.data_path,
        args.batch,
        args.max_candidates,
        args.max_len,
        tokenizer,
        args.num_worker,
        args.indicate_mention_boundary,
    )

    model = Zeshel(encoder)
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=args.epoch
        * num_train_examples
        / (args.batch * args.accumulate_gradient_steps),
        num_warmup_steps=args.warm_up_proportion
        * args.epoch
        * num_train_examples
        / (args.batch * args.accumulate_gradient_steps),
    )
    # eps=args.adam_epsilon ? why

    model.zero_grad()
    num_steps = 0
    loss_value = 0
    for epoch_num in range(args.epoch):
        for i, batch in enumerate(train_loader):
            loss = model(batch[0], batch[1], batch[2], batch[3])["loss"]
            loss.backward()
            loss_value += loss.item()

            if i + 1 % args.accumulate_gradient_steps == 0:
                optimizer.step()
                scheduler.step()
                print("the loss for step {} is {}".format(num_steps, loss_value))
                loss_value = 0
                model.zero_grad()
                loss_value = 0
                num_steps += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data/zeshel/zeshel_dev")
    parser.add_argument("--max_candidates", default=10)
    parser.add_argument("--max_len", default=256)
    parser.add_argument("--num_worker", default=1)
    parser.add_argument("--indicate_mention_boundary", default=True)
    parser.add_argument("--lr", default=0.05)
    parser.add_argument("--adam_epsilon", default=1e-8)
    parser.add_argument("--accumulate_gradient_steps", default=1)
    parser.add_argument("--warm_up_proportion", default=0.1)
    parser.add_argument("--batch", default=50)
    parser.add_argument("--epoch", default=50)

    args = parser.parse_args()

    main()
