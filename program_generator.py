import sys
import random
import string
import io
from contextlib import redirect_stdout

from data import encode_sequences, dense_to_one_hot


GO_SYMBOL = 'G'
PAD_SYMBOL = '_'
LETTERS = string.digits + string.ascii_lowercase + ' +-=():\n'
SYMBOLS = [PAD_SYMBOL, GO_SYMBOL] + list(LETTERS)
SYMBOL_TO_IDX = dict((l, i) for i, l in enumerate(SYMBOLS))

INPUT_SEQ_LEN = 32
OUTPUT_SEQ_LEN = 10
MAX_NUM_LEN = 2


class ProgramGenerator():

    def __init__(self, batch_size, number_len=2):
        self.number_len = number_len
        self.batch_size = batch_size

    def random_digit(self):
        return random.randint(0, 10 ** random.randint(1, self.number_len))

    def generate_program(self):
        program = \
            ["def f():",
             "  print({0}+{1})".format(self.random_digit(), self.random_digit()),
             "f()"]
        return "\n".join(program)

    def increase_difficulty(self):
        if self.number_len < MAX_NUM_LEN:
            self.number_len += 1

    def has_max_difficulty(self):
        return self.number_len == MAX_NUM_LEN

    def next_batch(self):

        programs = [self.generate_program() for _ in range(self.batch_size)]

        # Execute the programs to get the target
        results = []
        for program in programs:
            with io.StringIO() as buf, redirect_stdout(buf):
                exec(program)
                results.append(buf.getvalue()[:-1])

        input_sequences = encode_sequences(programs,
                                           symbol_to_idx=SYMBOL_TO_IDX,
                                           sequence_len=INPUT_SEQ_LEN,
                                           go_symbol=GO_SYMBOL,
                                           pad_symbol=PAD_SYMBOL,
                                           pad_beginning=True)
        input_sequences = dense_to_one_hot(input_sequences,
                                           num_classes=len(SYMBOL_TO_IDX))

        target_sequences = encode_sequences(results,
                                            symbol_to_idx=SYMBOL_TO_IDX,
                                            sequence_len=OUTPUT_SEQ_LEN,
                                            go_symbol=None,
                                            pad_beginning=False,
                                            pad_symbol=PAD_SYMBOL)
        target_sequences = dense_to_one_hot(target_sequences,
                                            num_classes=len(SYMBOL_TO_IDX))

        return input_sequences, target_sequences
