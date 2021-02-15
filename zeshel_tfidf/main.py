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


def construct_optimizer(args, model, num_train_examples):
    no_weight_decay = ["LayerNorm.weight", "bias"]
    optimized_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(np in n for np in no_weight_decay)
            ],
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if all(np not in n for np in no_weight_decay)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = AdamW(optimized_parameters, lr=args.lr, eps=args.adam_epsilon)

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

    return optimizer, scheduler


def construct_optimizer_simple(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_constant_schedule(model.parameters())

    return optimizer, scheduler


print("here")


def eval(model, loader, num_examples):
    model.eval()
    num_correct = 0
    num_total = 0
    for i, batch in enumerate(loader):
        prediction = model(
            batch[0].to(args.device),
            batch[1].to(args.device),
            batch[2].to(args.device),
            batch[3].to(args.device),
        )["predictions"]
        num_correct += torch.sum(
            prediction == torch.zeros(prediction.size(0)).to(args.device)
        )
        num_total += prediction.size(0)
    accuracy = (num_correct / num_examples) * 100

    return {"accuracy": accuracy, "num_correct": num_correct, "num_total": num_total}


def load_model(args, eval_mode):
    checkpoint = torch.load(args.model)

    encoder = BertModel.from_pretrained("bert-base-uncased")

    model = Zeshel(encoder).to(args.device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if eval_mode:
        model.eval()

    return model


def main(args):
    set_seed()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder = BertModel.from_pretrained("bert-base-uncased")

    data = load_zeshel_data(args.data_path)

    num_train_examples = len(data[1])
    num_val_examples = len(data[3])
    num_test_examples = len(data[4])

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

    if args.complex_optimizer:
        optimizer, scheduler = construct_optimizer(args, model, num_train_examples)
    else:
        optimizer, scheduler = construct_optimizer_simple(args, model)

    model.zero_grad()
    num_steps = 0
    loss_value = 0
    best_val_perf = 0.0
    model = model.to(args.device)
    for epoch_num in range(args.epoch):
        for i, batch in enumerate(train_loader):
            model.train()
            loss = model(
                batch[0].to(args.device),
                batch[1].to(args.device),
                batch[2].to(args.device),
                batch[3].to(args.device),
            )["loss"]
            loss.backward()
            loss_value += loss.item()
            print("the loss is {}".format(loss))

            if (i + 1) % args.accumulate_gradient_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                num_steps += 1
        loss_value = 0

        train_eval_result = eval(model, train_loader, num_train_examples)
        val_eval_result = eval(model, train_loader, num_val_examples)
        print("training accuracy is {}".format(train_eval_result["accuracy"]))
        print("validation accuracy is {}".format(val_eval_result["accuracy"]))

        if val_eval_result["accuracy"] > best_val_perf:
            torch.save(
                {
                    "epoch": epoch_num,
                    "model_state_dict": model.state_dict(),
                    "hyperparameters": args,
                },
                args.model,
            )

    # load and test
    model = load_model(args, eval_mode=True)
    test_result = eval(model, test_loader, num_test_examples)
    print("test accuracy is {}".format(test_result["accuracy"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_model.pt")
    parser.add_argument("--data_path", default="../data/zeshel/zeshel_dev")
    parser.add_argument("--max_candidates", default=64)
    parser.add_argument("--max_len", default=256)
    parser.add_argument("--num_worker", default=1)
    parser.add_argument("--indicate_mention_boundary", default=True)

    parser.add_argument("--lr", default=0.05)
    parser.add_argument("--weight_decay", default=0.01)
    parser.add_argument("--adam_epsilon", default=1e-8)
    parser.add_argument("--accumulate_gradient_steps", default=1)
    parser.add_argument("--clip", default=1)
    parser.add_argument("--warm_up_proportion", default=0.1)
    parser.add_argument("--batch", default=1)
    parser.add_argument("--epoch", default=3)
    parser.add_argument("--complex_optimizer", default=True)

    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    main(args)
