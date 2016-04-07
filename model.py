import time
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
from tensorflow.models.rnn import seq2seq, rnn, rnn_cell

DEFAULT_LEARNING_RATE = 0.01


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
                 is_training=False):

        self.session = session
        self.batch_size = batch_size
        self.num_input_symbols = num_input_symbols
        self.num_output_symbols = num_output_symbols

        self.encoder_inputs = []
        self.decoder_inputs = []

        for i in range(input_sequence_len - 1):
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=(None, num_input_symbols),
                                                      name="encoder_{0}".format(i)))

        for i in range(output_sequence_len + 1):
            self.decoder_inputs.append(tf.placeholder(tf.float32, shape=(None, num_output_symbols),
                                                      name="decoder_{0}".format(i)))

        if num_layers > 1:
            cells = [rnn_cell.BasicLSTMCell(hidden_units, input_size=num_input_symbols)]
            cells += [rnn_cell.BasicLSTMCell(hidden_units, input_size=hidden_units) for _ in range(num_layers - 1)]
            self.cell = rnn_cell.MultiRNNCell(cells)
        else:
            self.cell = rnn_cell.BasicLSTMCell(hidden_units, input_size=num_input_symbols)

        self.w_softmax = tf.get_variable('w_softmax', shape=(hidden_units, num_output_symbols),
                                         initializer=tf.random_normal_initializer())
        self.b_softmax = tf.get_variable('b_softmax', shape=(num_output_symbols,),
                                         initializer=tf.random_normal_initializer())

        # seq2seq_outputs is a list of arrays with output_sequence_len: [(batch_size x hidden_units)]
        seq2seq_outputs, _ = self._init_seq2seq(self.encoder_inputs, self.decoder_inputs, self.cell,
                                                feed_previous=not is_training)

        output_logits = [tf.matmul(seq2seq_output, self.w_softmax) + self.b_softmax
                         for seq2seq_output in seq2seq_outputs]
        self.output_probs = [tf.nn.softmax(logit) for logit in output_logits]

        if is_training:
            self.targets = self.decoder_inputs[1:]
            losses = [tf.nn.softmax_cross_entropy_with_logits(logit, target)
                      for logit, target in zip(output_logits, self.targets)]
            loss = tf.reduce_sum(tf.add_n(losses))

            self.cost = loss / output_sequence_len / self.batch_size
            self.learning_rate = tf.Variable(DEFAULT_LEARNING_RATE, trainable=False)

            train_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_vars), 5.0)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, train_vars))

    def _init_seq2seq(self, encoder_inputs, decoder_inputs, cell, feed_previous):

        def inference_loop_function(prev, _):
            prev = tf.nn.xw_plus_b(prev, self.w_softmax, self.b_softmax)
            return tf.to_float(tf.equal(prev, tf.reduce_max(prev, reduction_indices=[1], keep_dims=True)))

        loop_function = inference_loop_function if feed_previous else None

        with variable_scope.variable_scope('seq2seq'):
            _, enc_state = rnn.rnn(cell, encoder_inputs, dtype=dtypes.float32)
            return seq2seq.rnn_decoder(decoder_inputs, enc_state, cell, loop_function=loop_function)

    def set_learning_rate(self, learning_rate):
        self.session.run(tf.assign(self.learning_rate, learning_rate))
        # self.learning_rate.assign(learning_rate).run()

    def init_variables(self):
        tf.initialize_all_variables().run()

    def _fit_batch(self, input_values, targets):
        assert targets.shape[0] == input_values.shape[0] == self.batch_size
        assert len(self.encoder_inputs) == input_values.shape[1] - 1
        assert len(self.decoder_inputs) == targets.shape[1] + 1

        input_feed = {}
        for i, encoder_input in enumerate(self.encoder_inputs):
            input_feed[encoder_input.name] = input_values[:, i, :]

        # The last time step of the input should be the GO symbol
        input_feed[self.decoder_inputs[0].name] = input_values[:, -i, :]

        for i, decoder_input in enumerate(self.decoder_inputs[1:]):
            input_feed[decoder_input.name] = targets[:, i]

        train_loss, _ = self.session.run([self.cost, self.train_op], input_feed)

        return train_loss

    def fit(self,
            data_generator,
            num_epochs=10,
            batches_per_epoch=256,
            lr_decay=0.8):

        prev_loss = np.inf

        for e in range(num_epochs):
            for b in range(batches_per_epoch):
                start = time.time()

                inputs, targets = next(data_generator)
                train_loss = self._fit_batch(inputs, targets)

                end = time.time()

            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                  .format(b, batches_per_epoch, e, train_loss, end - start))

            if train_loss >= prev_loss:
                self.set_learning_rate(self.learning_rate.eval() * lr_decay)
                print("Decreasing LR to {:.5f}".format(self.learning_rate.eval()))

            prev_loss = train_loss

    def fit_curr(self,
                 data_generator,
                 testing_model,
                 num_epochs=30,
                 batches_per_epoch=256,
                 lr_decay=0.8,
                 num_val_batches=128):

        prev_error_rate = np.inf
        val_error_rate = np.inf

        val_set = [data_generator.next_batch() for _ in range(num_val_batches)]

        for e in range(num_epochs):

            start = time.time()
            for b in range(batches_per_epoch):
                inputs, targets = data_generator.next_batch()
                train_loss = self._fit_batch(inputs, targets)
            end = time.time()

            val_error_rate = testing_model.validate(val_set)

            print("Epoch {}: train_loss = {:.3f}, val_error_rate = {:.3f}, time/epoch = {:.3f}"
                  .format(e, train_loss, val_error_rate, end - start))

            if val_error_rate >= prev_error_rate and data_generator.has_max_difficulty():
                self.set_learning_rate(self.learning_rate.eval() * lr_decay)
                print("Decreasing LR to {:.5f}".format(self.learning_rate.eval()))

            elif val_error_rate < 0.05:
                print("Increasing difficulty")
                data_generator.increase_difficulty()
                val_set = [data_generator.next_batch() for _ in range(num_val_batches)]

            prev_error_rate = val_error_rate

    def predict(self, encoder_input_values, go_symbol_idx=0):

        num_batches = encoder_input_values.shape[0]

        input_feed = {}
        for i, encoder_input in enumerate(self.encoder_inputs):
            input_feed[encoder_input.name] = encoder_input_values[:, i, :]

        # Set all decoder inputs to the GO symbol. However, all but the first will be ignored
        decoder_input_value = np.zeros((num_batches, self.num_output_symbols), dtype=np.float32)
        decoder_input_value[:, go_symbol_idx] = 1.0

        for decoder_input in self.decoder_inputs:
            input_feed[decoder_input.name] = decoder_input_value

        symbol_probs = self.session.run(self.output_probs, input_feed)
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
