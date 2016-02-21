import numpy as np


def dense_to_one_hot(labels_dense, num_classes=10):
    """
    Convert class labels from scalars to one-hot vectors.

    Args:
        labels_dense: array, 1D or 2D, int32
            Shape: (num_samples) or (num_sequences, sequence_len)
        num_classes: int

    Returns:
        labels_one_hot: array, 2D or 3D, float32
            Shape: (num_samples, num_classes) or
            (num_sequences, sequence_len, num_classes)
    """

    assert labels_dense.ndim == 1 or labels_dense.ndim == 2
    assert labels_dense.dtype == np.int32

    if labels_dense.ndim == 1:
        num_sequences = 0
        sequence_len = labels_dense.shape
    else:
        num_sequences, sequence_len = labels_dense.shape

    labels_dense = labels_dense.reshape(-1)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    if num_sequences > 0:
        labels_one_hot = labels_one_hot.reshape((num_sequences, sequence_len, num_classes))

    return labels_one_hot
