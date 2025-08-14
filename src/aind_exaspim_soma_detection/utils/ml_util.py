"""
Created on Fri Jan 3 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training and inference.

"""

from random import sample
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import numpy as np
import torch

from aind_exaspim_soma_detection.machine_learning.models import FastConvNet3d


def load_model(path, patch_shape, device="cuda"):
    """
    Loads a pre-trained model from the given, then transfers the model to the
    specified device (i.e. CPU or GPU).

    Parameters
    ----------
    path : str
        Path to the saved model weights.
    patch_shape : Tuple[int]
        Shape of the input patches expected by the model expects.
    device : str, optional
        Name of device where model should be loaded and run. The default is
        "cuda".

    Returns
    -------
    FastConvNet3d
        Model instance with the loaded weights.
    """
    model = FastConvNet3d(patch_shape)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    return model


def get_correct(keys, y, hat_y, threshold, verbose=True):
    """
    Identifies true positives and true negatives from a prediction.

    Parameters
    ----------
    keys : List[tuple]
        Identifiers corresponding to each example.
    y : List[int]
        Ground truth labels for each example.
    hat_y : list of float
        Predicted likelihoods for each example.
    threshold : float
        Threshold value for classifying the predicted likelihoods.
    verbose : bool, optional
        Indication of whether to print our the number of true positives and
        true negatives. The default is True.

    Returns
    -------
    dict
        A dictionary with the keys "true_negatives" and "true_positives" and
        values that consist of tuples containing a key and predicted value.
    """
    # Extract incorrect
    correct = {"true_negatives": list(), "true_positives": list()}
    for i, (y_i, hat_y_i) in enumerate(zip(y, hat_y)):
        if y_i == 0 and hat_y_i < threshold:
            correct["true_negatives"].append((keys[i], hat_y_i))
        elif y_i == 1 and hat_y_i > threshold:
            correct["true_positives"].append((keys[i], hat_y_i))

    # Report results
    if verbose:
        n_true_negatives = len(correct["true_negatives"])
        n_true_positives = len(correct["true_positives"])
        print(f"# True Positives: {n_true_positives}")
        print(f"# True Negatives: {n_true_negatives}")
    return correct


def get_incorrect(keys, y, hat_y, threshold, verbose=True):
    """
    Identifies false positives and false negatives from a prediction.

    Parameters
    ----------
    keys : List[tuple]
        Identifiers corresponding to each example.
    y : List[int]
        Ground truth labels for each example.
    hat_y : list of float
        Predicted likelihoods for each example.
    threshold : float
        Threshold value for classifying the predicted likelihoods.
    verbose : bool, optional
        Indication of whether to print our the number of false positives and
        false negatives. The default is True.

    Returns
    -------
    dict
        A dictionary with the keys "false_negatives" and "false_positives" and
        values that consist of tuples containing a key and predicted value.
    """
    # Extract incorrect
    incorrect = {"false_negatives": list(), "false_positives": list()}
    for i, (y_i, hat_y_i) in enumerate(zip(y, hat_y)):
        if y_i == 1 and hat_y_i < threshold:
            incorrect["false_negatives"].append((keys[i], hat_y_i))
        elif y_i == 0 and hat_y_i > threshold:
            incorrect["false_positives"].append((keys[i], hat_y_i))

    # Report results
    if verbose:
        n_false_negatives = len(incorrect["false_negatives"])
        n_false_positives = len(incorrect["false_positives"])
        print(f"# False Positives: {n_false_positives}")
        print(f"# False Negatives: {n_false_negatives}")
    return incorrect


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
