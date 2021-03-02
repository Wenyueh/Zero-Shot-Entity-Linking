import unittest
from evaluate import macro_averaged_evaluate
import numpy as np


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.predictions = [
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 1, 1, 1]),
            np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
            np.array([0, 0]),
        ]
        self.targets = [
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0]),
        ]
        self.accuracy = 0.6

    def test_macro_averaged_evaluate(self):
        result = macro_averaged_evaluate(self.predictions, self.targets)
        self.assertEqual(result, self.accuracy, "Macro averaged evaluation wrong")


if __name__ == "__main__":
    unittest.main()
