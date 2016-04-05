import random
import string

from data import encode_sequences, dense_to_one_hot


GO_SYMBOL = 'G'
PAD_SYMBOL = '_'
INPUT_LETTERS = string.digits + '+'
SYMBOLS = [PAD_SYMBOL, GO_SYMBOL] + list(INPUT_LETTERS)
SYMBOL_TO_IDX = dict((l, i) for i, l in enumerate(SYMBOLS))

MAX_NUM_LEN = 5
INPUT_SEQ_LEN = MAX_NUM_LEN * 2 + 3
OUTPUT_SEQ_LEN = MAX_NUM_LEN + 2


class AdditionGenerator():

    def __init__(self, batch_size, number_len=2):
        self.number_len = number_len
        self.batch_size = batch_size

    def random_digit(self):
        return random.randint(0, 10 ** random.randint(1, self.number_len))

    def increase_difficulty(self):
        if self.number_len < MAX_NUM_LEN:
            self.number_len += 1

    def has_max_difficulty(self):
        return self.number_len == MAX_NUM_LEN

    def next_batch(self):

        ints_batch = [(self.random_digit(),
                       self.random_digit()) for _ in range(self.batch_size)]
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

        return input_sequences, target_sequences
