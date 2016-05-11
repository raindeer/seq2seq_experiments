import tensorflow as tf

from program_generator import ProgramGenerator, SYMBOL_TO_IDX, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN
# from addition_generator import AdditionGenerator, SYMBOL_TO_IDX, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN
from model import Seq2SeqModel

hidden_units = 320
num_layers = 2
num_symbols = len(SYMBOL_TO_IDX)
num_epochs = 20000
training_batch_size = 128
batches_per_epoch = 128
num_val_batches = 256

data_generator = ProgramGenerator(batch_size=training_batch_size, program_length=1, num_len=1)
# data_generator = AdditionGenerator(batch_size=training_batch_size)

with tf.device('/gpu:0'):
    with tf.Session() as session:

        print("Building model")
        model = Seq2SeqModel(session=session,
                             hidden_units=hidden_units,
                             num_layers=num_layers,
                             input_sequence_len=INPUT_SEQ_LEN,
                             output_sequence_len=OUTPUT_SEQ_LEN,
                             num_input_symbols=num_symbols,
                             num_output_symbols=num_symbols,
                             batch_size=training_batch_size,
                             is_training=True,
                             scope='model')

        model.init_variables()

        print("Finished building model")

        model.fit(data_generator,
                  num_epochs=num_epochs,
                  batches_per_epoch=batches_per_epoch,
                  num_val_batches=num_val_batches)

        print("Finished training")
