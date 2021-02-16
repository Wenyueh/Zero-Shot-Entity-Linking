#!/usr/bin/env ipython

import torch


def evaluate(predictions):
    num_domains = len(predictions)
    averaged_accuracy = 0
    for prediction in predictions:
        averaged_accuracy += (
            torch.sum(prediction == torch.zeros(len(prediction)).to(prediction.device))
        ) / len(prediction)
    return averaged_accuracy / num_domains
