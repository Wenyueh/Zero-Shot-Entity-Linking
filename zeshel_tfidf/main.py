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
from dataloader import get_loaders, load_zeshel_data, ZeshelDataset
from torch.utils.data import DataLoader
from macroaveeval import macro_averaged_evaluate

import random
import numpy as np
from collections import OrderedDict


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


def return_predictions_targets(args, model, tokenizer, data, sample_data):
    model.eval()

    categories = []
    for mc in sample_data:
        if mc[0]["corpus"] not in categories:
            categories.append(mc[0]["corpus"])

    predictions = []
    targets = []
    for cat in categories:
        domain_sample = []
        for mc in sample_data:
            if mc[0]["corpus"] == cat:
                domain_sample.append(mc)
        dataset = ZeshelDataset(
            data[0],
            domain_sample,
            args.max_candidates,
            args.max_len,
            tokenizer,
            False,
            True,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch, num_workers=args.num_worker, shuffle=False,
        )
        prediction = torch.tensor([]).to(args.device)
        for i, batch in enumerate(dataloader):
            batch_prediction = model(
                batch[0].to(args.device),
                batch[1].to(args.device),
                batch[2].to(args.device),
                batch[3].to(args.device),
            )["predictions"]
            prediction = torch.cat((prediction, batch_prediction), dim=0)
        predictions.append(prediction.tolist())
        targets.append(torch.zeros(prediction.size()).tolist())

    return predictions, targets


def load_model(args, dp, eval_mode):
    checkpoint = torch.load(args.model)

    if dp:
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            k = k[7:]
            new_state_dict[k] = v
        checkpoint["model_state_dict"] = new_state_dict

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
    dp = torch.cuda.device_count() > 1
    if dp:
        model = nn.DataParallel(model)
    model = model.to(args.device)

    if args.complex_optimizer:
        optimizer, scheduler = construct_optimizer(args, model, num_train_examples)
    else:
        optimizer, scheduler = construct_optimizer_simple(args, model)

    model.zero_grad()
    num_steps = 0
    loss_value = 0
    best_val_perf = 0.0
    for epoch_num in range(args.epoch):
        for i, batch in enumerate(train_loader):
            model.train()
            loss = model(
                batch[0].to(args.device),
                batch[1].to(args.device),
                batch[2].to(args.device),
                batch[3].to(args.device),
            )["loss"]
            if dp:
                loss = torch.mean(loss, dim=0)
            loss.backward()
            loss_value += loss.item()

            if (i + 1) % args.accumulate_gradient_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                num_steps += 1
        loss_value = 0

        train_predictions, train_targets = return_predictions_targets(
            args, model, tokenizer, data, data[1]
        )
        train_accuracy = macro_averaged_evaluate(train_predictions, train_targets)
        val_predictions, val_targets = return_predictions_targets(
            args, model, tokenizer, data, data[3]
        )
        val_accuracy = macro_averaged_evaluate(val_predictions, val_targets)
        print(
            "for epoch {}, training accuracy is {}, validation accuracy is {}".format(
                epoch_num, train_accuracy, val_accuracy
            )
        )

        if val_accuracy > best_val_perf:
            torch.save(
                {
                    "epoch": epoch_num,
                    "model_state_dict": model.state_dict(),
                    "hyperparameters": args,
                },
                args.model,
            )

    # load and test
    model = load_model(args, dp, eval_mode=True)
    test_predictions, test_targets = return_predictions_targets(
        args, model, tokenizer, data, data[4]
    )
    test_accuracy = macro_averaged_evaluate(test_predictions, test_targets)
    print("test accuracy is {}".format(test_accuracy))


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
    parser.add_argument("--batch", default=2)
    parser.add_argument("--epoch", default=5)
    parser.add_argument("--complex_optimizer", default=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", type=str, default="0")

    args = parser.parse_args()

    main(args)
