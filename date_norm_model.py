import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.models.rnn import seq2seq, rnn, rnn_cell
from tensorflow.python.ops import variable_scope

from date_generator import generate_training_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class DateNormModel():

    def __init__(self,
                 session,
                 hidden_units=128,
                 input_sequence_len=20,
                 output_sequence_len=10,
                 num_input_symbols=20,
                 num_output_symbols=10,
                 batch_size=32,
                 is_training=False):

        self.session = session
        self.batch_size = batch_size
        self.num_input_symbols = num_input_symbols
        self.num_output_symbols = num_output_symbols

        self.encoder_inputs = []
        self.decoder_inputs = []

        for i in range(input_sequence_len):
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=(batch_size, num_input_symbols),
                                                      name="encoder_{0}".format(i)))
        if is_training:
            for i in range(output_sequence_len + 1):
                self.decoder_inputs.append(tf.placeholder(tf.float32, shape=(batch_size, num_output_symbols),
                                                          name="decoder_{0}".format(i)))
            self.targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]
        else:
            self.decoder_inputs.append(tf.placeholder(tf.float32, shape=(batch_size, num_output_symbols),
                                                      name="decoder_0"))

        cell = rnn_cell.BasicLSTMCell(hidden_units, input_size=num_input_symbols)

        self.w_softmax = weight_variable([hidden_units, num_output_symbols])
        self.b_softmax = bias_variable([num_output_symbols])

        seq2seq_outputs, _ = self._seq2seq(self.encoder_inputs, self.decoder_inputs, cell,
                                           feed_previous=not is_training)
        # outputs: [batch_size x hidden_units], list of output_sequence_len

        output_logits = [tf.matmul(seq2seq_output, self.w_softmax) + self.b_softmax
                         for seq2seq_output in seq2seq_outputs]
        self.output_probs = [tf.nn.softmax(logit) for logit in output_logits]

        #output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        #self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        #self.probs = tf.nn.softmax(self.logits)

        if is_training:
            losses = [tf.nn.softmax_cross_entropy_with_logits(logit, target)
                      for logit, target in zip(output_logits, self.targets)]
            loss = tf.reduce_sum(tf.add_n(losses))

            self.cost = loss / output_sequence_len / self.batch_size

            self.learning_rate = tf.Variable(0.0, trainable=False)
            train_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_vars), 5.0)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, train_vars))

    def _seq2seq(self, encoder_inputs, decoder_inputs, cell, feed_previous):

        def inference_loop_function(prev, _):
            prev = tf.nn.xw_plus_b(prev, self.w_softmax, self.b_softmax)
            return tf.to_float(tf.equal(prev, tf.reduce_max(prev, reduction_indices=[1], keep_dims=True)))

        loop_function = inference_loop_function if feed_previous else None

        with variable_scope.variable_scope('seq2seq'):
            _, enc_state = rnn.rnn(cell, encoder_inputs, dtype=dtypes.float32)
            return seq2seq.rnn_decoder(decoder_inputs, enc_state, cell,
                                       loop_function=loop_function)

    def fit(self, encoder_inputs, targets):
        assert targets.shape[0] == encoder_inputs.shape[0] == self.batch_size
        assert len(self.decoder_inputs) == targets.shape[1] + 1

        input_feed = {}
        for i in range(encoder_inputs.shape[1]):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[:, i]

        for i in range(targets.shape[1]):
            input_feed[self.decoder_inputs[i].name] = targets[:, i]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_decoder_input = self.decoder_inputs[targets.shape[1]].name
        input_feed[last_decoder_input] = np.zeros([targets.shape[0], targets.shape[2]], dtype=np.int32)

        train_loss, _ = self.session.run([self.cost, self.train_op], input_feed)

        return train_loss

    def predict(self, encoder_inputs, go_symbol_idx=0):

        print("encoder_inputs", encoder_inputs.shape)

        input_feed = {}
        for i in range(encoder_inputs.shape[1]):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[:, i, :]

        decoder_input = np.zeros((encoder_inputs.shape[0], self.num_output_symbols),
                                 dtype=np.float32)
        decoder_input[:, go_symbol_idx] = 1.0

        input_feed[self.decoder_inputs[0].name] = decoder_input

        symbol_probs = self.session.run(self.output_probs, input_feed)

        print(decoder_input)
        print(len(symbol_probs), symbol_probs[0].shape)

        return symbol_probs

    def set_learning_rate(self, learning_rate=0.01):
        self.session.run(tf.assign(self.learning_rate, learning_rate))

    def init_variables(self):
        tf.initialize_all_variables().run()


def train(model,
          num_epochs=10,
          batches_per_epoch=1024,
          batch_size=32):

    date_iterator = generate_training_data(batch_size=batch_size)

    model.init_variables()
    model.set_learning_rate(0.001)

    # s aver = tf.train.Saver(tf.all_variables())

    for e in range(num_epochs):
        # sess.run(tf.assign(model.learning_rate, args.learning_rate * (args.decay_rate ** e)))

        for b in range(batches_per_epoch):
            start = time.time()

            encoder_inputs, targets = next(date_iterator)

            train_loss = model.fit(encoder_inputs, targets)
            end = time.time()

        print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
              .format(b, batches_per_epoch, e, train_loss, end - start))

        # if (e * data_loader.num_batches + b) % args.save_every == 0:
        #    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
        #    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
        #    print "model saved to {}".format(checkpoint_path)


def decode_output_sequences(sequences, symbols):
    decoded_sequences = []
    for sequence in np.argmax(sequences, axis=2):
        decoded_sequences.append(''.join(symbols[idx] for idx in sequence))
    return decoded_sequences
