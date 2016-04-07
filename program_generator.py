import string
import io
from contextlib import redirect_stdout

from random import choice, random, randint
from data import encode_sequences, dense_to_one_hot


GO_SYMBOL = 'G'
PAD_SYMBOL = '_'
LETTERS = string.digits + string.ascii_lowercase + ' +-=():<>\n'
SYMBOLS = [PAD_SYMBOL, GO_SYMBOL] + list(LETTERS)
SYMBOL_TO_IDX = dict((l, i) for i, l in enumerate(SYMBOLS))

INPUT_SEQ_LEN = 100
OUTPUT_SEQ_LEN = 10

MAX_NUM_LENGTH = 3
MAX_PROGRAM_LENGTH = 4

COMPARATORS = ('<', '>')
OPERATORS = ('+', '-')
VARIABLE_NAMES = list('abcdefgh')


def if_operation(variables, nesting, difficulty):
    compare_variable = choice(list(variables))
    comparator = choice(COMPARATORS)
    compare_value = random_digit(difficulty)
    change_variable = choice(list(variables))
    change_value = random_digit(difficulty)
    operator = choice(OPERATORS)
    code = 'if {0}{1}{2}:'.format(compare_variable,
                                  comparator,
                                  compare_value,
                                  change_variable,
                                  operator,
                                  change_value)
    nesting += 1
    return code, nesting


def assign_operation(variables, nesting, num_len):

    variable = choice(VARIABLE_NAMES)
    variables.add(variable)
    value = random_digit(num_len)
    code = '{0}={1}'.format(variable, value)
    return code, nesting


def add_or_sub_operation(variables, nesting, num_len):

    variable = choice(list(variables))
    operator = choice(OPERATORS)
    value = random_digit(num_len)
    if random() < 0.5:
        code = '{0}{1}={2}'.format(variable, operator, value)
    else:
        variable2 = choice(list(variables))
        code = '{0}={1}{2}{3}'.format(variable, variable2, operator, value)

    return code, nesting


def print_operation(variables, nesting, num_len):
    operator = choice(OPERATORS)
    code = 'print({0})'.format(operator.join(list(variables)))
    return code, nesting

OPERATIONS = (if_operation, add_or_sub_operation)


def generate_program(num_len, length):
    variables = set()
    nesting = 0

    lines = []
    num_lines = randint(1, length)
    code, _ = assign_operation(variables, nesting, num_len)
    lines.append(code)

    for i in range(num_lines):
        if nesting == 0:
            operation = choice(OPERATIONS + (assign_operation,))
        else:
            operation = choice(OPERATIONS)

        code, new_nesting = operation(variables, nesting, num_len)
        lines.append(''.join(['  '] * nesting) + code)
        if nesting == new_nesting and random() < 0.5:
            nesting -= 1
        nesting = new_nesting

    if nesting > 0:
        code, new_nesting = add_or_sub_operation(variables, nesting, num_len)
        lines.append(''.join(['  '] * nesting) + code)

    code, _ = print_operation(variables, nesting, num_len)
    lines.append(code)

    return '\n'.join(lines)


def random_digit(difficulty):
    return randint(0, 10 ** randint(1, difficulty))


class ProgramGenerator():

    def __init__(self, batch_size, program_length=2, num_len=2):
        self.program_length = program_length
        self.num_len = num_len
        self.num_len_was_last_inc = False
        self.batch_size = batch_size

    def generate_program(self):
        return generate_program(num_len=self.num_len,
                                length=self.program_length)

    def increase_difficulty(self):

        if self.has_max_difficulty():
            return

        # Alternate between increasing program length and number length
        if self.num_len_was_last_inc:
            self.program_length += 1
            self.num_len_was_last_inc = False
        else:
            self.num_len += 1
            self.num_len_was_last_inc = True

        print("Difficulty:", self.num_len, self.program_length)

    def has_max_difficulty(self):
        return self.num_len >= MAX_NUM_LENGTH and \
            self.program_length >= MAX_PROGRAM_LENGTH

    def next_batch(self):

        programs = [self.generate_program() for _ in range(self.batch_size)]

        # Execute the programs to get the targets
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
