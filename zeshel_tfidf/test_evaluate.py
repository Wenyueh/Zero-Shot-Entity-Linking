import unittest
from evaluate import macro_averaged_evaluate


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.predictions = [
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0],
        ]
        self.targets = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0],
        ]
        self.accuracy = 0.6

    def test_macro_averaged_evaluate(self):
        result = macro_averaged_evaluate(self.predictions, self.targets)
        self.assertEqual(result, self.accuracy, "Macro averaged evaluation wrong")


if __name__ == "__main__":
    unittest.main()
