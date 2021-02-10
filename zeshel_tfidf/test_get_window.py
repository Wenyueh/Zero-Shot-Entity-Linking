#!/usr/bin/env ipython

from dataloader import get_window
import unittest


class test_get_window(unittest.TestCase):
    def setUP(self):
        pass

    def test_long_mention(self):
        window, start, end = get_window([1, 2, 3, 4, 5], [0, 1, 0], [1, 0, 1], 4)
        self.assertEqual(
            window,
            [1, 2, 3, 4],
            "window wrong with max length even, window longer than max length",
        )
        self.assertEqual(
            [start, end],
            [0, 3],
            "start position wrong with max length even, window longer than max length",
        )

        window, start, end = get_window([1, 2, 3, 4, 5], [0, 1, 0], [1, 0, 1], 5)
        self.assertEqual(
            window,
            [1, 2, 3, 4, 5],
            "window wrong with max length odd, window longer than max length",
        )
        self.assertEqual(
            [start, end],
            [0, 4],
            "start or end position wrong with max length odd, window longer than max length",
        )

    def test_long_prefix_long_suffix(self):
        window, start, end = get_window([1, 2], [0, 1, 0], [1, 0, 1], 4)
        self.assertEqual(
            window,
            [0, 1, 2, 1],
            "window wrong with max length even, long prefix, long suffix",
        )
        self.assertEqual(
            [start, end],
            [1, 2],
            "start or end position wrong with max length even, long prefix, long suffix",
        )

        window, start, end = get_window([1, 2], [0, 1, 0], [1, 0, 1], 5)
        self.assertEqual(
            window,
            [1, 0, 1, 2, 1],
            "window wrong with max length odd, long prefix, long suffix",
        )
        self.assertEqual(
            [start, end],
            [2, 3],
            "start or end position wrong with max length odd, long prefix, long suffix",
        )

    def test_long_prefix_short_suffix(self):
        window, start, end = get_window([1, 2], [0, 1, 0, 1, 0], [1], 6)
        self.assertEqual(
            window,
            [0, 1, 0, 1, 2, 1],
            "window wrong with max length even, long prefix, short suffix",
        )
        self.assertEqual(
            [start, end],
            [3, 4],
            "start or end position wrong with max length even, long prefix, short suffix",
        )

        window, start, end = get_window([1, 2], [0, 1, 0, 1, 0], [1], 7)
        self.assertEqual(
            window,
            [1, 0, 1, 0, 1, 2, 1],
            "window wrong with max length odd, long prefix, short suffix",
        )
        self.assertEqual(
            [start, end],
            [4, 5],
            "start or end position wrong with max length odd, long prefix, short suffix",
        )

    def test_short_prefix_long_suffix(self):
        window, start, end = get_window([1, 2], [0, 1], [1, 0, 1, 0, 1], 6)
        self.assertEqual(
            window,
            [0, 1, 1, 2, 1, 0],
            "window wrong with max length even, short prefix, long suffix",
        )
        self.assertEqual(
            [start, end],
            [2, 3],
            "start or end position wrong with max length even, short prefix, long suffix",
        )

        window, start, end = get_window([1, 2], [0, 1], [1, 0, 1, 0, 1], 7)
        self.assertEqual(
            window,
            [0, 1, 1, 2, 1, 0, 1],
            "window wrong with max length odd, short prefix, long suffix",
        )
        self.assertEqual(
            [start, end],
            [2, 3],
            "start or end position wrong with max length odd, short prefix, long suffix",
        )

    def test_short_prefix_short_suffix(self):
        window, start, end = get_window([1, 2], [0, 1], [1], 6)
        self.assertEqual(
            window,
            [0, 1, 1, 2, 1],
            "window wrong with max length even, short prefix, short suffix",
        )
        self.assertEqual(
            [start, end],
            [2, 3],
            "start or end position wrong with max length even, short prefix, short suffix",
        )

        window, start, end = get_window([1, 2], [0, 1], [1, 0], 7)
        self.assertEqual(
            window,
            [0, 1, 1, 2, 1, 0],
            "window wrong with max length odd, short prefix, short suffix",
        )
        self.assertEqual(
            [start, end],
            [2, 3],
            "start or end position wrong with max length odd, short prefix, short suffix",
        )


if __name__ == "__main__":
    unittest.main()
