import numpy as np
import numpy as np


def macro_averaged_evaluate(predictions, targets):

    assert isinstance(predictions, list)
    assert isinstance(predictions, list)

    num_domains = len(predictions)
    averaged_accuracy = 0
    for prediction, target in zip(predictions, targets):
        assert isinstance(prediction, np.ndarray)
        assert isinstance(target, np.ndarray)
        averaged_accuracy += (sum(prediction == target)) / len(prediction)

    return averaged_accuracy / num_domains
