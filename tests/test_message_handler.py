"""Tests aind_library_template printers methods."""

import doctest
import unittest

import pandas as pd

from aind_library_template import message_handlers
from aind_library_template.message_handlers import MessageHandler


class MessageHandlerTest(unittest.TestCase):
    """Tests MessageHandler methods."""

    @classmethod
    def setUpClass(cls):
        """Set up the Test class with basic attributes that can be shared."""
        my_msg = "Hello World!!"
        cls.my_msg = my_msg
        cls.msg_handler = MessageHandler(my_msg)

    def test_to_screaming_snake_case(self):
        """Tests _to_screaming_snake_case method."""
        input_string = "! a random message 012 %"
        expected_output = "A_RANDOM_MESSAGE_012"
        actual_output = self.msg_handler._to_screaming_snake_case(input_string)
        self.assertEqual(expected_output, actual_output)

    def test_log_msg(self):
        """Tests that the log_msg method logs a message."""
        with self.assertLogs() as captured:
            self.msg_handler.log_msg()

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), self.my_msg)

    def test_msg_as_df(self):
        """Tests that the message gets returned as a pandas DataFrame."""

        # df from msg with default col_name
        df1 = self.msg_handler.msg_as_df()
        # df from msg with non-default col_name
        df2 = self.msg_handler.msg_as_df(col_name="non_default")

        # Expected outputs
        expected_df1 = pd.DataFrame.from_dict({"message": [self.my_msg]})
        expected_df2 = pd.DataFrame.from_dict({"non_default": [self.my_msg]})

        self.assertTrue(df1.equals(expected_df1))
        self.assertTrue(df2.equals(expected_df2))
        self.assertTrue(not df1.equals(expected_df2))


# It's possible to run doctests the same time unit tests are run by adding this
def load_tests(_loader, tests, _ignore):
    """Add this to run doctests in module"""
    tests.addTests(doctest.DocTestSuite(message_handlers))
    return tests


if __name__ == "__main__":
    unittest.main()
