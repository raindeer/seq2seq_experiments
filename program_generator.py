import string
import io
from contextlib import redirect_stdout

from random import choice, random, randint
from data import encode_sequences, dense_to_one_hot

GO_SYMBOL = 'G'
PAD_SYMBOL = '_'
LETTERS = string.digits + string.ascii_lowercase + ' +-=():<>\n'
SYMBOLS = [GO_SYMBOL, PAD_SYMBOL] + list(LETTERS)
#SYMBOLS = [PAD_SYMBOL, GO_SYMBOL] + list(LETTERS)
SYMBOL_TO_IDX = dict((l, i) for i, l in enumerate(SYMBOLS))

INPUT_SEQ_LEN = 70
OUTPUT_SEQ_LEN = 6

MAX_NUM_LENGTH = 2
MAX_PROGRAM_LENGTH = 3

COMPARATORS = ('<', '>')
OPERATORS = ('+', '-')
VARIABLE_NAMES = list('abcdefgh')


def if_operation(variables, nesting, difficulty):
    compare_variable = choice(list(variables))
    comparator = choice(COMPARATORS)
    compare_value = random_digit(difficulty)
    #change_variable = choice(list(variables))
    #change_value = random_digit(difficulty)
    #operator = choice(OPERATORS)
    code = 'if {0}{1}{2}:'.format(compare_variable,
                                  comparator,
                                  compare_value)
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


OPERATIONS = (add_or_sub_operation, if_operation, assign_operation)


def generate_program(num_len, length):
    variables = set()
    nesting = 0

    lines = []
    lines.append(assign_operation(variables, nesting, num_len)[0])

    if length > 0:
        num_lines = randint(1, length)
        for i in range(num_lines):
            if num_lines <= 1:
                operation = add_or_sub_operation
            elif nesting == 0:
                operation = choice(OPERATIONS)
            else:
                operation = choice((add_or_sub_operation, if_operation))

            code, new_nesting = operation(variables, nesting, num_len)
            lines.append(''.join(['  '] * nesting) + code)
            if nesting == new_nesting and random() < 0.5:
                nesting -= 1
            nesting = new_nesting

        if nesting > 0:
            code, new_nesting = add_or_sub_operation(variables, nesting, num_len)
            lines.append(''.join(['  '] * nesting) + code)

    lines.append(print_operation(variables, nesting, num_len)[0])

    return '\n'.join(lines)


def random_digit(difficulty):
    size = 10 ** randint(1, difficulty)
    if difficulty > 1:
        return randint(-size, size)
    else:
        return randint(0, size)


class ProgramGenerator():
    def __init__(self, batch_size, program_length=1, num_len=1):
        self.program_length = program_length
        self.num_len = num_len
        self.num_len_was_last_inc = False
        self.batch_size = batch_size

    def generate_program(self, hash_mod=None):

        if hash_mod is None:
            program = generate_program(num_len=self.num_len,
                                       length=self.program_length)
        else:
            program_hash = None
            while program_hash != hash_mod:
                program = generate_program(num_len=self.num_len,
                                           length=self.program_length)
                program_hash = hash(program) % 2
        return program

    def increase_difficulty(self):

        if self.has_max_difficulty():
            return

        # Alternate between increasing program length and number length
        if self.num_len_was_last_inc:
            if self.num_len < MAX_PROGRAM_LENGTH:
                self.program_length += 1
            self.num_len_was_last_inc = False
        else:
            if self.num_len < MAX_NUM_LENGTH:
                self.num_len += 1
            self.num_len_was_last_inc = True

        print("Difficulty:", self.num_len, self.program_length)

    def has_max_difficulty(self):
        return self.num_len >= MAX_NUM_LENGTH and \
               self.program_length >= MAX_PROGRAM_LENGTH

    def difficulty(self):
        return (self.num_len, self.program_length)

    def next_batch(self, validation=False):

        programs = [self.generate_program(hash_mod=0 if validation else 1)
                    for _ in range(self.batch_size)]

        # Execute the programs to get the targets
        results = []
        for program in programs:
            with io.StringIO() as buf, redirect_stdout(buf):
                exec(program)
                results.append(buf.getvalue()[:-1])

        input_sequences = encode_sequences(programs,
                                           symbol_to_idx=SYMBOL_TO_IDX,
                                           sequence_len=INPUT_SEQ_LEN,
                                           #go_symbol=GO_SYMBOL,
                                           pad_symbol=PAD_SYMBOL,
                                           pad_beginning=True,
                                           reverse=True)
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
