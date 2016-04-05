import random
import string

from data import encode_sequences, dense_to_one_hot


GO_SYMBOL = 'G'
PAD_SYMBOL = '_'
LETTERS = string.digits + string.ascii_lowercase + '+-=():\n'
SYMBOLS = [PAD_SYMBOL, GO_SYMBOL] + list(LETTERS)
SYMBOL_TO_IDX = dict((l, i) for i, l in enumerate(SYMBOLS))

INPUT_SEQ_LEN = 50
OUTPUT_SEQ_LEN = 10


def random_digit(max_len):
    return random.randint(0, 10 ** random.randint(1, max_len))


def generate_program():
    return "print({0}+{1})".format(random_digit(2), random_digit(2))


def generate_data(batch_size=32):

    while True:
        programs = [generate_program() for _ in range(batch_size)]
        results = [eval(program) for program in programs]

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

        yield input_sequences, target_sequences
