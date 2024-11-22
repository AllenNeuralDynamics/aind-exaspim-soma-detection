""" Example module for a basic class with methods and doctests. We use Numpy
docstring format."""

import logging
import re

import pandas as pd


class MessageHandler:
    """
    Small class to handle messages.
    """

    def __init__(self, msg: str):
        """
        Initialize class with a msg
        Parameters
        ----------
        msg : str
        """
        self.msg = msg

    @staticmethod
    def _to_screaming_snake_case(input_str: str) -> str:
        """
        Converts a string to SCREAMING_SNAKE_CASE
        >>> MessageHandler._to_screaming_snake_case("hello world!")
        'HELLO_WORLD'

        Parameters
        ----------
        input_str : str

        Returns
        -------
        str
          The input_str in SCREAMING_SNAKE_CASE format

        """
        new_str = re.sub(r"[^\w\s]", "", input_str).strip()
        new_str = re.sub(r"(\s|_|-)+", "_", new_str.upper())
        return new_str

    def log_msg(self):
        """Simply logs the message."""
        logging.info(self.msg)

    def msg_as_df(self, col_name: str = "message") -> pd.DataFrame:
        """
        Returns message as a dataframe.
        >>> msg_handler = MessageHandler('hello world')
        >>> msg_handler.msg_as_df(col_name='msg')
                   msg
        0  hello world

        Parameters
        ----------
        col_name : str
          Custom column name for the pandas data frame. 'message' by default.

        Returns
        -------
        pd.DataFrame

        """
        return pd.DataFrame.from_dict({col_name: [self.msg]})
