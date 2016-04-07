import tensorflow as tf

from program_generator import ProgramGenerator, SYMBOL_TO_IDX, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN
from model import Seq2SeqModel


hidden_units = 64
num_layers = 2
training_batch_size = 32
num_symbols = len(SYMBOL_TO_IDX)
num_epochs = 20000
batches_per_epoch = 16

program_generator = ProgramGenerator(batch_size=training_batch_size)

with tf.Session() as session:

    print("Building model")

    with tf.variable_scope('model', reuse=None):
        training_model = Seq2SeqModel(session=session,
                                      hidden_units=hidden_units,
                                      num_layers=num_layers,
                                      input_sequence_len=INPUT_SEQ_LEN,
                                      output_sequence_len=OUTPUT_SEQ_LEN,
                                      num_input_symbols=num_symbols,
                                      num_output_symbols=num_symbols,
                                      batch_size=training_batch_size,
                                      is_training=True)

    training_model.init_variables()

    with tf.variable_scope('model', reuse=True):
        testing_model = Seq2SeqModel(session=session,
                                     hidden_units=hidden_units,
                                     num_layers=num_layers,
                                     input_sequence_len=INPUT_SEQ_LEN,
                                     output_sequence_len=OUTPUT_SEQ_LEN,
                                     num_input_symbols=num_symbols,
                                     num_output_symbols=num_symbols,
                                     batch_size=training_batch_size,
                                     is_training=False)

    print("Finished building model")

    training_model.fit_curr(program_generator,
                            testing_model=testing_model,
                            num_epochs=num_epochs,
                            batches_per_epoch=batches_per_epoch)

    print("Finished training")
