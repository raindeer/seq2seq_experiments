import random
import string
from datetime import datetime
from datetime import timedelta

from data import encode_sequences, dense_to_one_hot

START_DATE = datetime.strptime('1950-01-01', '%Y-%m-%d')
END_DATE = datetime.strptime('2050-12-31', '%Y-%m-%d')

FORMAT_TOKENS = ('%d', '%b', '%B', '%m', '%y', '%Y')
INPUT_FORMATS = ['%Y %B, %d',
                 '%B %d, %Y',
                 '%b %d %y',
                 '%Y-%m-%d',
                 '%Y/%m/%d',
                 '%y-%m-%d',
                 '%d-%m-%y']

OUTPUT_FORMAT = '%Y-%m-%d'

GO_SYMBOL = 'G'
PAD_SYMBOL = '_'
LETTERS = string.ascii_lowercase + string.digits + ',- .:/'
SYMBOLS = [GO_SYMBOL, PAD_SYMBOL] + list(LETTERS)
SYMBOL_TO_IDX = dict((l, i) for i, l in enumerate(SYMBOLS))

INPUT_SEQ_LEN = 20
OUTPUT_SEQ_LEN = 12 + 1


def random_datetime(start=START_DATE, end=END_DATE):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)


def format_date(date, random_format=False):
    if random_format:
        random_format_tokens = random.sample(FORMAT_TOKENS, random.randint(3, 5))
        date_format = '-'.join(random_format_tokens)
    else:
        date_format = random.choice(INPUT_FORMATS)
    return date.strftime(date_format).lower()


def generate_data(batch_size=32, random_format=False):

    while True:

        datetimes = [random_datetime() for _ in range(batch_size)]
        input_date_strings = [format_date(dt, random_format=random_format) for dt in datetimes]
        target_date_strings = [dt.strftime(OUTPUT_FORMAT) for dt in datetimes]

        input_sequences = encode_sequences(input_date_strings,
                                           symbol_to_idx=SYMBOL_TO_IDX,
                                           sequence_len=INPUT_SEQ_LEN,
                                           pad_symbol=PAD_SYMBOL,
                                           go_symbol=GO_SYMBOL,
                                           pad_beginning=True,
                                           reverse=True)
        input_sequences = dense_to_one_hot(input_sequences,
                                           num_classes=len(SYMBOL_TO_IDX))

        target_sequences = encode_sequences(target_date_strings,
                                            symbol_to_idx=SYMBOL_TO_IDX,
                                            sequence_len=OUTPUT_SEQ_LEN,
                                            pad_symbol=PAD_SYMBOL,
                                            go_symbol=None,
                                            pad_beginning=False)
        target_sequences = dense_to_one_hot(target_sequences,
                                            num_classes=len(SYMBOL_TO_IDX))

        yield input_sequences, target_sequences
