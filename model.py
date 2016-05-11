import time
import numpy as np
import tensorflow as tf
import json

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
from tensorflow.models.rnn import seq2seq, rnn, rnn_cell

DEFAULT_LEARNING_RATE = 0.01

from data import decode_output_sequences


class Seq2SeqGraph():
    def __init__(self,
                 is_training=False,
                 hidden_units=128,
                 num_layers=1,
                 input_sequence_len=20,
                 output_sequence_len=10,
                 num_input_symbols=20,
                 num_output_symbols=20,
                 weight_amplitude=0.08,
                 batch_size=32,
                 peep=False):

        self.encoder_inputs = []
        self.decoder_inputs = []

        for i in range(input_sequence_len):
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=(None, num_input_symbols),
                                                      name="encoder_{0}".format(i)))

        for i in range(output_sequence_len + 1):
            self.decoder_inputs.append(tf.placeholder(tf.float32, shape=(None, num_output_symbols),
                                                      name="decoder_{0}".format(i)))

        def random_uniform():
            return tf.random_uniform_initializer(-weight_amplitude, weight_amplitude)

        if num_layers > 1:
            cells = [rnn_cell.LSTMCell(hidden_units, use_peepholes=peep, input_size=num_input_symbols,
                                       initializer=random_uniform())]
            cells += [rnn_cell.LSTMCell(hidden_units, use_peepholes=peep, input_size=hidden_units,
                                        initializer=random_uniform()) for _ in range(num_layers - 1)]
            self.cell = rnn_cell.MultiRNNCell(cells)
        else:
            self.cell = rnn_cell.LSTMCell(hidden_units, use_peepholes=peep,
                                          initializer=random_uniform())

        self.w_softmax = tf.get_variable('w_softmax', shape=(hidden_units, num_output_symbols),
                                         initializer=random_uniform())
        self.b_softmax = tf.get_variable('b_softmax', shape=(num_output_symbols,),
                                         initializer=random_uniform())

        # decoder_outputs is a list of tensors with output_sequence_len: [(batch_size x hidden_units)]
        decoder_outputs, _ = self._init_seq2seq(self.encoder_inputs, self.decoder_inputs, self.cell,
                                                feed_previous=not is_training)

        output_logits = [tf.matmul(decoder_output, self.w_softmax) + self.b_softmax
                         for decoder_output in decoder_outputs]
        self.output_probs = [tf.nn.softmax(logit) for logit in output_logits]

        # If this is a training model create the training operation and loss function
        if is_training:
            self.targets = self.decoder_inputs[1:]
            losses = [tf.nn.softmax_cross_entropy_with_logits(logit, target)
                      for logit, target in zip(output_logits, self.targets)]

            loss = tf.reduce_sum(tf.add_n(losses))
            self.cost = loss / output_sequence_len / batch_size
            self.learning_rate = tf.Variable(DEFAULT_LEARNING_RATE, trainable=False)

            train_vars = tf.trainable_variables()
            grads = tf.gradients(self.cost, train_vars)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            self.train_op = optimizer.apply_gradients(zip(grads, train_vars))

    def _init_seq2seq(self, encoder_inputs, decoder_inputs, cell, feed_previous):

        def inference_loop_function(prev, _):
            prev = tf.nn.xw_plus_b(prev, self.w_softmax, self.b_softmax)
            return tf.to_float(tf.equal(prev, tf.reduce_max(prev, reduction_indices=[1], keep_dims=True)))

        loop_function = inference_loop_function if feed_previous else None

        with variable_scope.variable_scope('seq2seq'):
            _, final_enc_state = rnn.rnn(cell, encoder_inputs, dtype=dtypes.float32)
            return seq2seq.rnn_decoder(decoder_inputs, final_enc_state, cell, loop_function=loop_function)


class Seq2SeqModel:
    def __init__(self,
                 session,
                 hidden_units=128,
                 num_layers=1,
                 input_sequence_len=20,
                 output_sequence_len=10,
                 num_input_symbols=20,
                 num_output_symbols=20,
                 batch_size=32,
                 go_symbol_idx=0,
                 symbols=None,
                 scope='seq2seq_model'):

        self.session = session
        self.batch_size = batch_size
        self.symbols = symbols
        self.go_decoder_input_value = np.zeros((batch_size, num_output_symbols), dtype=np.float32)
        self.go_decoder_input_value[:, go_symbol_idx] = 1.0

        # We need to creat two different graphs one where the output of the decoder is looped back
        # to the decoder input (inference) and one where the decoder input is set to the targets (training
        with tf.variable_scope(scope, reuse=None):
            self.training_graph = Seq2SeqGraph(hidden_units=hidden_units,
                                               num_layers=num_layers,
                                               input_sequence_len=input_sequence_len,
                                               output_sequence_len=output_sequence_len,
                                               num_input_symbols=num_input_symbols,
                                               num_output_symbols=num_output_symbols,
                                               batch_size=batch_size,
                                               is_training=True)

        with tf.variable_scope(scope, reuse=True):
            self.testing_graph = Seq2SeqGraph(hidden_units=hidden_units,
                                              num_layers=num_layers,
                                              input_sequence_len=input_sequence_len,
                                              output_sequence_len=output_sequence_len,
                                              num_input_symbols=num_input_symbols,
                                              num_output_symbols=num_output_symbols,
                                              batch_size=batch_size,
                                              is_training=False)

    def set_learning_rate(self, learning_rate):
        self.session.run(tf.assign(self.training_graph.learning_rate, learning_rate))

    def get_learning_rate(self):
        return self.training_graph.learning_rate.eval()

    def init_variables(self):
        tf.initialize_all_variables().run()

    def _fit_batch(self, input_values, targets):
        assert targets.shape[0] == input_values.shape[0] == self.batch_size
        assert len(self.training_graph.encoder_inputs) == input_values.shape[1]
        assert len(self.training_graph.decoder_inputs) == targets.shape[1] + 1

        input_feed = {}
        for i, encoder_input in enumerate(self.training_graph.encoder_inputs):
            input_feed[encoder_input.name] = input_values[:, i, :]

        # The first input of the decoder is the padding symbol (we use the same symbol for GO and PAD)
        input_feed[self.training_graph.decoder_inputs[0].name] = self.go_decoder_input_value

        for i, decoder_input in enumerate(self.training_graph.decoder_inputs[1:]):
            input_feed[decoder_input.name] = targets[:, i]

        train_loss, _ = self.session.run([self.training_graph.cost,
                                          self.training_graph.train_op], feed_dict=input_feed)

        return train_loss

    def fit(self,
            data_generator,
            num_epochs=30,
            batches_per_epoch=256,
            lr_decay=0.95,
            num_val_batches=128,
            output_dir='output'):

        with tf.device('/cpu:0'):
            saver = tf.train.Saver()

        history = []
        prev_error_rate = np.inf
        val_error_rate = np.inf
        best_val_error_rate = np.inf

        val_set = [data_generator.next_batch(validation=True) for _ in range(num_val_batches)]

        epochs_since_init = 0

        for e in range(num_epochs):

            if self.symbols:
                self.examples(data_generator)

            start = time.time()
            for b in range(batches_per_epoch):
                inputs, targets = data_generator.next_batch(validation=False)
                train_loss = self._fit_batch(inputs, targets)

            end = time.time()

            val_error_rate = self.validate(val_set)

            if epochs_since_init > 15 and epochs_since_init < 17 and val_error_rate > 0.85:
                self.init_variables()
                epochs_since_init = 0
                print("Restarting...")
                continue
            epochs_since_init += 1

            print("Epoch {}: train_loss = {:.3f}, val_error_rate = {:.3f}, time/epoch = {:.3f}, diff: {}"
                  .format(e, train_loss, val_error_rate, end - start, data_generator.difficulty()))

            if best_val_error_rate > val_error_rate:
                save_path = saver.save(self.session, "{}/model_{}.ckpt".format(output_dir,
                                                                               data_generator.difficulty()))
                print("Model saved in file: %s" % save_path)
                best_val_error_rate = val_error_rate

            if val_error_rate > prev_error_rate and data_generator.has_max_difficulty():
                self.set_learning_rate(self.get_learning_rate() * lr_decay)
                print("Decreasing LR to {:.5f}".format(self.get_learning_rate()))

            elif val_error_rate < 0.10:
                print("Increasing difficulty")
                if not data_generator.has_max_difficulty():
                    data_generator.increase_difficulty()
                    best_val_error_rate = np.inf
                val_set = [data_generator.next_batch() for _ in range(num_val_batches)]

            history.append({
                'val_error_rate': float(val_error_rate),
                'train_loss': float(train_loss),
                'learning_rate': float(self.get_learning_rate()),
                'difficulty': data_generator.difficulty()
            })

            with open('{}/history.json'.format(output_dir), 'w') as outfile:
                json.dump(history, outfile)

            prev_error_rate = val_error_rate

    def predict(self, encoder_input_values, pad_symbol_idx=0):

        input_feed = {}
        for i, encoder_input in enumerate(self.testing_graph.encoder_inputs):
            input_feed[encoder_input.name] = encoder_input_values[:, i, :]

        for decoder_input in self.testing_graph.decoder_inputs:
            input_feed[decoder_input.name] = self.go_decoder_input_value

        symbol_probs = self.session.run(self.testing_graph.output_probs, input_feed)
        symbol_probs = np.array(symbol_probs)
        symbol_probs = np.transpose(symbol_probs, (1, 0, 2))

        return symbol_probs

    def validate(self, val_set):

        num_correct = 0
        num_samples = 0

        for batch in val_set:
            x, y = batch
            target = np.argmax(y, axis=2)
            prediction = np.argmax(self.predict(x), axis=2)[:, :-1]

            num_correct += sum([int(np.all(t == p)) for t, p in zip(target, prediction)])
            num_samples += len(x)

        return 1.0 - float(num_correct) / num_samples

    def load(self, checkpoint_file):
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_file)

    def examples(self, data_generator, num_examples=5):
        """
        Prints some examples during training
        Args:
            data_generator:

        """
        assert self.symbols

        x, y = data_generator.next_batch(validation=True)

        # input_strings = decode_output_sequences(x, symbols=SYMBOLS)
        target_strings = decode_output_sequences(y, symbols=self.symbols)

        model_output = self.predict(x)
        pred_strings = decode_output_sequences(model_output, symbols=self.symbols)

        print(target_strings[:num_examples])
        print(pred_strings[:num_examples])
