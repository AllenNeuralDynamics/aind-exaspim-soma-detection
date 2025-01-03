"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training and inference.

"""


def split_train_validation(examples, train_ratio):
    """
    Splits a dictionary of examples into training and validation sets based on
    a given ratio.

    Parameters
    ----------
    examples : dict
        A dictionary where keys represent example identifiers, and values
        represent the example data.
    train_ratio : float
        Number between 0 and 1 representing the proportion of the dataset to be
        used for training.

    Returns
    -------
    Tuple[dict]
        A tuple containing two dictionaries:
            - Dictionary containing the training examples.
            - Dictionary containing the validation examples.

    """
    # Get numbers of examples
    n_train_examples = int(train_ratio * len(examples))
    n_valid_examples = len(examples) - n_train_examples

    # Sample keys
    train_keys = sample(examples.keys(), n_train_examples)
    valid_keys = examples.keys() - train_keys

    # Get examples
    train_examples = dict({k: examples[k] for k in train_keys})
    valid_examples = dict({k: examples[k] for k in valid_keys})
    return train_examples, valid_examples
