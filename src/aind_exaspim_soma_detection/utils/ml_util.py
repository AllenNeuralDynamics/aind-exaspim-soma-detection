"""
Created on Fri Jan 3 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training and inference.

"""

from random import sample
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

import numpy as np


def split_train_validation(examples, train_ratio=0.85):
    """
    Splits a dictionary of examples into training and validation sets based on
    a given ratio.

    Parameters
    ----------
    examples : dict
        A dictionary where keys represent example identifiers, and values
        represent the example data.
    train_ratio : float, optional
        Number between 0 and 1 representing the proportion of the dataset to be
        used for training. The default is 0.85.

    Returns
    -------
    Tuple[dict]
        A tuple containing two dictionaries:
            - Dictionary containing the training examples.
            - Dictionary containing the validation examples.

    """
    # Sample keys
    n_train_examples = int(train_ratio * len(examples))
    train_keys = sample(examples.keys(), n_train_examples)
    valid_keys = examples.keys() - train_keys

    # Get examples
    train_examples = dict({k: examples[k] for k in train_keys})
    valid_examples = dict({k: examples[k] for k in valid_keys})
    return train_examples, valid_examples


def report_metrics(y, hat_y):
    """
    Computes and prints various evaluation metrics based on the true labels
    "y" and predicted labels "hat_y".

    Parameters
    ----------
    y : numpy.ndarray
        True labels.
    hat_y : numpy.ndarray
        Predicted labels from a machine learning model.

    Returns
    -------
    None

    """
    print("Accuracy:", accuracy_score(y, hat_y))
    print("Accuracy Dif:", accuracy_score(y, hat_y) - np.sum(y) / len(y))
    print("Precision:", precision_score(y, hat_y))
    print("Recall:", recall_score(y, hat_y))
    print("F1:", f1_score(y, hat_y))
    print("")
