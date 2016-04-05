import numpy as np


def encode_sequences(letter_sequences, symbol_to_idx, sequence_len, pad_symbol=None, go_symbol=None,
                     pad_beginning=True, reverse=False, ):
    """
    Given a set of symbols and their index/label encoded the given
    list of string sequences as numeric sequences.
    """

    pad_idx = symbol_to_idx[pad_symbol]

    if go_symbol is None:
        go_idx = None
    else:
        go_idx = symbol_to_idx[go_symbol]

    assert sequence_len >= len(max(letter_sequences, key=len)) + 0 if go_idx is None else 1

    encoded_sequences = np.full((len(letter_sequences), sequence_len),
                                fill_value=pad_idx,
                                dtype=np.int32)

    for i, sequence in enumerate(letter_sequences):

        idxs = [symbol_to_idx[symbol] for symbol in sequence]

        if reverse:
            idxs = idxs[::-1]

        # Insert the idx of the GO symbol to the end of the sequence.
        if go_idx is not None:
            idxs.append(go_idx)

        if pad_beginning:
            encoded_sequences[i, -len(idxs):] = idxs
        else:
            encoded_sequences[i, :len(idxs)] = idxs

    return encoded_sequences


def decode_output_sequences(sequences, symbols):
    """
    Args:
        sequences: ndarray
            Shape: (num_seq, time_steps, output_size)
        symbols: [str]

    Returns:
        decoded_sequences: [str]
    """

    decoded_sequences = []
    for sequence in np.argmax(sequences, axis=2):
        decoded_sequences.append(''.join(symbols[idx] for idx in sequence))
    return decoded_sequences


def dense_to_one_hot(labels_dense, num_classes):
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
