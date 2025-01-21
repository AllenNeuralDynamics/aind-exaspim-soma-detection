"""
Created on Fri Jan 3 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training and inference.

"""

import numpy as np
from random import sample
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


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


def report_metrics(y, hat_y, threshold):
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
    hat_y = (hat_y > threshold).astype(int)
    accuracy = accuracy_score(y, hat_y)
    print("Accuracy:", round(accuracy, 4))
    print("Accuracy Dif:", round(accuracy - np.sum(y) / len(y), 4))
    print("Precision:", round(precision_score(y, hat_y), 4))
    print("Recall:", round(recall_score(y, hat_y), 4))
    print("F1:", round(f1_score(y, hat_y), 4))
    print("")


def toCPU(tensor, return_numpy=True):
    """
    Transfers a PyTorch tensor from the GPU to the CPU and optionally converts
    it to a NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor that is on a GPU.
    return_numpy : bool, optional
        Indication of whether to return the tensor as a NumPy array. The
        default is True.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Input tensor as a NumPy on the CPU if "return_numpy" is True.
        Otherwise, input tensor on the CPU.

    """
    tensor = tensor.detach().cpu()
    return np.array(tensor) if return_numpy else tensor
