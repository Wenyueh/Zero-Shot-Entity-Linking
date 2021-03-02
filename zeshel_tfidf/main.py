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
from dataloader import get_loaders, load_zeshel_data, ZeshelDataset, Logger
from torch.utils.data import DataLoader
from evaluate import macro_averaged_evaluate, micro_evaluate

import random
import math
import os
import time
import numpy as np
from collections import OrderedDict


def set_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


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
                if not any(np in n for np in no_weight_decay)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = AdamW(optimized_parameters, lr=args.lr, eps=args.adam_epsilon)

    num_training_steps = int(
        args.epoch * num_train_examples / (args.batch * args.accumulate_gradient_steps)
    )
    num_warmup_steps = int(args.warm_up_proportion * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_training_steps,
    )

    return optimizer, scheduler, num_training_steps, num_warmup_steps


def macro_dataloaders(args, tokenizer, data, sample_data):
    dataloader_list = []
    target_size_list = []

    categories = {}

    for mc in sample_data:
        if mc[0]["corpus"] not in categories:
            categories[mc[0]["corpus"]] = 1
        else:
            categories[mc[0]["corpus"]] += 1

    for cat in categories.keys():
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
        dataloader_list.append(dataloader)
        if not args.normalized:
            target_size_list.append(categories[cat])
        else:
            target_size_list.append(len(dataset))

    return dataloader_list, target_size_list


def return_predictions_targets(args, model, dataloader_list, target_size_list):
    model.eval()

    predictions = []
    targets = []

    for dataloader, target_size in zip(dataloader_list, target_size_list):
        prediction = torch.tensor([]).to(args.device)
        for i, batch in enumerate(dataloader):
            batch_prediction = model(
                batch[0].to(args.device),
                batch[1].to(args.device),
                batch[2].to(args.device),
                batch[3].to(args.device),
            )["predictions"]
            prediction = torch.cat((prediction, batch_prediction), dim=0)
        prediction = np.concatenate(
            (prediction.cpu().numpy(), np.array([1] * (target_size - len(prediction))))
        )
        predictions.append(prediction)
        targets.append(torch.zeros(len(prediction)).cpu().numpy())

    return predictions, targets


def load_model(args, dp, eval_mode):
    checkpoint = torch.load(args.model)

    encoder = BertModel.from_pretrained("bert-base-uncased")

    model = Zeshel(encoder)

    if dp:
        model = nn.DataParallel(model)

    model.to(args.device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if eval_mode:
        model.eval()

    return model


def count_parameters(model):
    number_parameters = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            number_parameters += torch.numel(p)

    return number_parameters


def main(args):
    set_seed(args)

    logger = Logger(args.model + ".log")
    logger.log(str(args))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder = BertModel.from_pretrained("bert-base-uncased")

    data = load_zeshel_data(args.data_path)

    num_train_examples = len(data[1])
    num_val_examples = len(data[4])
    num_test_examples = len(data[5])

    dp = torch.cuda.device_count() > 1

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
    if dp:
        model = nn.DataParallel(model)
    model = model.to(args.device)

    optimizer, scheduler, num_training_steps, num_warmup_steps = construct_optimizer(
        args, model, num_train_examples
    )

    logger.log("\n[Train]")
    logger.log("train sample size: {}".format(len(data[1])))
    logger.log("epoch number: {}".format(args.epoch))
    logger.log("batch size: {}".format(args.batch))
    logger.log("gradient accumulation steps: {}".format(args.accumulate_gradient_steps))
    logger.log(
        "effective batch size: {}".format(args.batch * args.accumulate_gradient_steps)
    )
    logger.log("train steps: {}".format(num_training_steps))
    logger.log("warmup steps: {}".format(num_warmup_steps))
    logger.log("learning rate: {}".format(args.lr))
    logger.log("parameter number: {}".format(count_parameters(model)))

    train_macro_dataloader_list, train_target_list = macro_dataloaders(
        args, tokenizer, data, data[1]
    )
    val_macro_dataloader_list, val_target_list = macro_dataloaders(
        args, tokenizer, data, data[4]
    )
    test_macro_dataloader_list, test_target_list = macro_dataloaders(
        args, tokenizer, data, data[5]
    )

    model.zero_grad()
    num_steps = 0
    loss_value = 0
    best_val_perf = float("-inf")
    for epoch_num in range(args.epoch):
        one_epoch_start_time = time.time()
        used_time = 0
        for i, batch in enumerate(train_loader):

            start_time = time.time()

            model.train()
            result = model(
                batch[0].to(args.device),
                batch[1].to(args.device),
                batch[2].to(args.device),
                batch[3].to(args.device),
            )
            loss = result["loss"]
            if dp:
                loss = torch.mean(loss)
            loss.backward()
            loss_value += loss.item()

            if (i + 1) % args.accumulate_gradient_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                num_steps += 1

                torch.cuda.synchronize()
                end_time = time.time()
                used_time += end_time - start_time

                if num_steps % args.logging_step == 0:
                    logger.log(
                        "From step {} to step {}, which is epoch {} batch {}, the total time used is {}, the averaged loss value is {}".format(
                            num_steps - args.logging_step,
                            num_steps,
                            epoch_num,
                            i,
                            used_time,
                            loss_value / args.logging_step,
                        )
                    )
                    used_time = 0
                    loss_value = 0

        train_predictions, train_targets = return_predictions_targets(
            args, model, train_macro_dataloader_list, train_target_list
        )
        val_predictions, val_targets = return_predictions_targets(
            args, model, val_macro_dataloader_list, val_target_list
        )
        if args.macro:
            train_accuracy = macro_averaged_evaluate(train_predictions, train_targets)
            val_accuracy = macro_averaged_evaluate(val_predictions, val_targets)
        else:
            train_accuracy = micro_evaluate(train_predictions, train_targets)
            val_accuracy = micro_evaluate(val_predictions, val_targets)
        logger.log(
            "for epoch {}, training accuracy is {}, validation accuracy is {}".format(
                epoch_num, train_accuracy, val_accuracy
            )
        )

        torch.cuda.synchronize()
        one_epoch_end_time = time.time()
        logger.log(
            "for epoch {}, training+evaluation time in total is {}".format(
                epoch_num, one_epoch_end_time - one_epoch_start_time
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
            logger.log(
                "for epoch {}, save model with validation accuracy {}".format(
                    epoch_num, val_accuracy
                )
            )
            best_val_perf = val_accuracy

    # load and test
    model = load_model(args, dp, eval_mode=True)
    val_predictions, val_targets = return_predictions_targets(
        args, model, val_macro_dataloader_list, val_target_list
    )
    print(len(val_predictions))
    print(len(val_targets))
    if args.macro:
        val_accuracy = macro_averaged_evaluate(val_predictions, val_targets)
    else:
        val_accuracy = micro_evaluate(val_predictions, val_targets)

    logger.log("test accuracy is {}".format(val_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=10)

    parser.add_argument("--model", type=str, default="best_model_full.pt")
    parser.add_argument("--data_path", type=str, default="../data/zeshel/zeshel_full")
    parser.add_argument("--max_candidates", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--indicate_mention_boundary", default=True)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--accumulate_gradient_steps", type=int, default=1)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--warm_up_proportion", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--complex_optimizer", default=True)

    parser.add_argument("--logging_step", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")

    parser.add_argument("--macro", type=bool, default=True)
    parser.add_argument("--normalized", type=bool, default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    main(args)
