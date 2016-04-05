import random
import string

from data import encode_sequences, dense_to_one_hot


GO_SYMBOL = 'G'
PAD_SYMBOL = '_'
INPUT_LETTERS = string.digits + '+'
SYMBOLS = [PAD_SYMBOL, GO_SYMBOL] + list(INPUT_LETTERS)
SYMBOL_TO_IDX = dict((l, i) for i, l in enumerate(SYMBOLS))

MAX_NUM_DIGITS = 5
INPUT_SEQ_LEN = MAX_NUM_DIGITS * 2 + 3
OUTPUT_SEQ_LEN = MAX_NUM_DIGITS + 2


def random_digit():
    return random.randint(0, 10 ** random.randint(1, MAX_NUM_DIGITS))


def generate_data(batch_size=32):

    while True:

        ints_batch = [(random_digit(), random_digit()) for _ in range(batch_size)]

        int_sum_batch = [sum(ints) for ints in ints_batch]
        addition_strings = ["{0}+{1}".format(*ints) for ints in ints_batch]
        sum_strings = [str(s) for s in int_sum_batch]

        input_sequences = encode_sequences(addition_strings,
                                           symbol_to_idx=SYMBOL_TO_IDX,
                                           sequence_len=INPUT_SEQ_LEN,
                                           go_symbol=GO_SYMBOL,
                                           pad_symbol=PAD_SYMBOL,
                                           pad_beginning=True,
                                           reverse=False)
        input_sequences = dense_to_one_hot(input_sequences,
                                           num_classes=len(SYMBOL_TO_IDX))

        target_sequences = encode_sequences(sum_strings,
                                            symbol_to_idx=SYMBOL_TO_IDX,
                                            sequence_len=OUTPUT_SEQ_LEN,
                                            go_symbol=None,
                                            pad_beginning=False,
                                            pad_symbol=PAD_SYMBOL)
        target_sequences = dense_to_one_hot(target_sequences,
                                            num_classes=len(SYMBOL_TO_IDX))

        yield input_sequences, target_sequences
