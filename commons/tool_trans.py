from header import *


def one_hot_to_dense(labels_one_hot):
    """
    convert one-hot encodings into labels, ensure the rank is 2

    Parameters
    ----------
    labels_one_hot : list of numbers presented by one hot encoding type.


    Returns
    -------
    np.darray


    See Also
    --------
    ref: https://www.kaggle.com/raoulma/mnist-image-class-tensorflow-cnn-99-51-test-acc/comments


    Examples
    --------
    >>> one_hot_to_dense(np.array([[1, 0, 0]]))
    [0]

    # todo
    judge labels_one_hot's shape
    """

    return np.argmax(labels_one_hot,1)


def dense_to_one_hot(labels_dense, num_classes):
    """
    convert class labels from scalars to one-hot vectors e.g. 1 => [0 1 0 0 0 0 0 0 0 0]

    Parameters
    ----------
    labels_dense : ndarray of int numbers
    num_classes : how many class

    Returns
    -------
    labels_one_hot : np.ndarray(np.array)

    See Also
    --------
    ref: https://www.kaggle.com/raoulma/mnist-image-class-tensorflow-cnn-99-51-test-acc/comments

    Examples
    --------
    >>> dense_to_one_hot(np.array([1, 2, 3]), 4)
    [[0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]


    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

