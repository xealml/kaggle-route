from header import *
from tool_trans import one_hot_to_dense


def accuracy_from_dense_labels(y_target, y_pred):
    """
    computet the accuracy of label predictions


    Parameters
    ----------
    y_target : the ground true
    y_pred : predict logits


    Returns
    -------
    accuracy: float


    See Also
    --------


    Examples
    --------

    """

    y_target = y_target.reshape(-1,)
    y_pred = y_pred.reshape(-1,)
    return np.mean(y_target == y_pred)


def accuracy_from_one_hot_labels(y_target, y_pred):
    """
    # computet the accuracy of one-hot encoded predictions

    Parameters
    ----------
    y_target : the ground true
    y_pred : predict logits


    Returns
    -------
    accuracy: float


    See Also
    --------


    Examples
    --------

    """

    y_target = one_hot_to_dense(y_target).reshape(-1,)
    y_pred = one_hot_to_dense(y_pred).reshape(-1,)
    return np.mean(y_target == y_pred)