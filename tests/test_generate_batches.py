"""Tests for batching the proposals whose metrics are computed."""

import unittest

from aind_exaspim_soma_detection.soma_proposal_classification import (
    generate_batches,
)


class GenerateBatchesTest(unittest.TestCase):
    """Every item is yielded once, in order, in batches of at most the size."""

    def test_every_item_is_batched(self):
        """No item is dropped, including the last one."""
        for n in (0, 1, 5, 63, 64, 65, 128, 130):
            with self.subTest(n=n):
                items = list(range(n))
                batches = list(generate_batches(items, 64))
                self.assertEqual([x for b in batches for x in b], items)
                self.assertTrue(all(0 < len(b) <= 64 for b in batches))


if __name__ == "__main__":
    unittest.main()
