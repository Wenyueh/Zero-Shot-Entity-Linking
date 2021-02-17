import numpy


def macro_averaged_evaluate(predictions, targets):
    assert type(predictions) == list
    assert type(targets) == list

    num_domains = len(predictions)
    averaged_accuracy = 0
    for prediction, target in zip(predictions, targets):
        averaged_accuracy += (
            sum(numpy.array(prediction) == numpy.array(target))
        ) / len(prediction)

    return averaged_accuracy / num_domains
