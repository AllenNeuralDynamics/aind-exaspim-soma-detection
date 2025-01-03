"""
Created on Fri Jan 3 12:30:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code used to train neural network to classify somas proposals.

"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluation_metrics(epoch, writer, y, hat_y, prefix=""):
    """
    Computes and logs various evaluation metrics to a TensorBoard.

    Parameters
    ----------
    epoch : int
        Current training epoch. Used as the x-axis value for logging metrics
        in the TensorBoard.
    writer : torch.utils.tensorboard.SummaryWriter
        TensorBoard writer object to log the metrics.
    y : ArrayLike
        True labels or ground truth values.
    hat_y : ArrayLike
        Predicted labels from a model.
    prefix : str, optional
        String prefix to prepend to the metric names when logging to
        TensorBoard. Default is an empty string.

    Returns
    -------
    float
        F1 score for the given epoch.

    """
    # Compute metrics
    accuracy = accuracy_score(y, hat_y)
    accuracy_dif = accuracy - np.sum(y) / len(y)
    f1 = f1_score(y, hat_y)
    precision = precision_score(y, hat_y)
    recall = recall_score(y, hat_y)

    # Write results to tensorboard
    writer.add_scalar(prefix + "_accuracy", accuracy, epoch)
    writer.add_scalar(prefix + "_accuracy_df", accuracy_dif, epoch)
    writer.add_scalar(prefix + "_precision:", precision, epoch)
    writer.add_scalar(prefix + "_recall:", recall, epoch)
    writer.add_scalar(prefix + "_f1:", f1, epoch)
    return f1
