import time
import numpy as np
import tensorflow as tf
import json

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
from tensorflow.models.rnn import seq2seq, rnn, rnn_cell

DEFAULT_LEARNING_RATE = 0.01

from data import decode_output_sequences
from program_generator import SYMBOLS

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
                 is_training=False):

        self.session = session
        self.batch_size = batch_size
        self.num_input_symbols = num_input_symbols
        self.num_output_symbols = num_output_symbols

        self.encoder_inputs = []
        self.decoder_inputs = []

        self.go_decoder_input_value = np.zeros((batch_size, num_output_symbols), dtype=np.float32)
        self.go_decoder_input_value[:, go_symbol_idx] = 1.0

        for i in range(input_sequence_len):
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=(None, num_input_symbols),
                                                      name="encoder_{0}".format(i)))

        for i in range(output_sequence_len + 1):
            self.decoder_inputs.append(tf.placeholder(tf.float32, shape=(None, num_output_symbols),
                                                      name="decoder_{0}".format(i)))

        peep = False
        a = 0.05
        if num_layers > 1:
            cells = [rnn_cell.LSTMCell(hidden_units, use_peepholes=peep, input_size=num_input_symbols,
                                       initializer=tf.random_uniform_initializer(-a, a))]
            cells += [rnn_cell.LSTMCell(hidden_units, use_peepholes=peep, input_size=hidden_units,
                                        initializer=tf.random_uniform_initializer(-a, a)) for _ in
                      range(num_layers - 1)]
            self.cell = rnn_cell.MultiRNNCell(cells)
        else:
            self.cell = rnn_cell.LSTMCell(hidden_units, use_peepholes=peep,
                                          initializer=tf.random_uniform_initializer(-a, a))


        a = 0.08
        self.w_softmax = tf.get_variable('w_softmax', shape=(hidden_units, num_output_symbols),
                                         initializer=tf.random_uniform_initializer(-a, a))
        self.b_softmax = tf.get_variable('b_softmax', shape=(num_output_symbols,),
                                         initializer=tf.random_uniform_initializer(-a, a)) #tf.constant_initializer(0.08)

        # seq2seq_outputs is a list of arrays with output_sequence_len: [(batch_size x hidden_units)]
        decoder_outputs, _ = self._init_seq2seq(self.encoder_inputs, self.decoder_inputs, self.cell,
                                                feed_previous=not is_training)

        output_logits = [tf.matmul(decoder_output, self.w_softmax) + self.b_softmax
                         for decoder_output in decoder_outputs]
        self.output_probs = [tf.nn.softmax(logit) for logit in output_logits]

        if is_training:
            self.targets = self.decoder_inputs[1:]
            losses = [tf.nn.softmax_cross_entropy_with_logits(logit, target)
                      for logit, target in zip(output_logits, self.targets)]

            loss = tf.reduce_sum(tf.add_n(losses))
            self.cost = loss / output_sequence_len / self.batch_size
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

    def set_learning_rate(self, learning_rate):
        self.session.run(tf.assign(self.learning_rate, learning_rate))

    def init_variables(self):
        tf.initialize_all_variables().run()

    def _fit_batch(self, input_values, targets):
        assert targets.shape[0] == input_values.shape[0] == self.batch_size
        assert len(self.encoder_inputs) == input_values.shape[1]
        assert len(self.decoder_inputs) == targets.shape[1] + 1

        input_feed = {}
        for i, encoder_input in enumerate(self.encoder_inputs):
            input_feed[encoder_input.name] = input_values[:, i, :]

        # The first input of the decoder is the padding symbol (we use the same symbol for GO and PAD)
        input_feed[self.decoder_inputs[0].name] = self.go_decoder_input_value

        for i, decoder_input in enumerate(self.decoder_inputs[1:]):
            input_feed[decoder_input.name] = targets[:, i]

        train_loss, _ = self.session.run([self.cost, self.train_op], feed_dict=input_feed)

        return train_loss

    def fit(self,
            data_generator,
            testing_model,
            num_epochs=30,
            batches_per_epoch=256,
            lr_decay=0.95,
            num_val_batches=128):

        with tf.device('/cpu:0'):
            saver = tf.train.Saver()

        history = []
        prev_error_rate = np.inf
        val_error_rate = np.inf
        best_val_error_rate = np.inf

        val_set = [data_generator.next_batch(validation=True) for _ in range(num_val_batches)]

        epochs_since_init = 0

        for e in range(num_epochs):

            testing_model.examples(data_generator)

            start = time.time()
            for b in range(batches_per_epoch):
                inputs, targets = data_generator.next_batch(validation=False)
                train_loss = self._fit_batch(inputs, targets)

            end = time.time()

            val_error_rate = testing_model.validate(val_set)

            if epochs_since_init > 15 and epochs_since_init < 17 and val_error_rate > 0.85:
                self.init_variables()
                epochs_since_init = 0
                print("Restarting...")
                continue
            epochs_since_init += 1

            print("Epoch {}: train_loss = {:.3f}, val_error_rate = {:.3f}, time/epoch = {:.3f}, diff: {}"
                  .format(e, train_loss, val_error_rate, end - start, data_generator.difficulty()))

            if best_val_error_rate > val_error_rate:
                save_path = saver.save(self.session, "output/model_{}.ckpt".format(data_generator.difficulty()))
                print("Model saved in file: %s" % save_path)
                best_val_error_rate = val_error_rate

            if val_error_rate > prev_error_rate and data_generator.has_max_difficulty():
                self.set_learning_rate(self.learning_rate.eval() * lr_decay)
                print("Decreasing LR to {:.5f}".format(self.learning_rate.eval()))

            elif val_error_rate < 0.10:
                print("Increasing difficulty")
                if not data_generator.has_max_difficulty():
                    data_generator.increase_difficulty()
                    best_val_error_rate = np.inf
                val_set = [data_generator.next_batch() for _ in range(num_val_batches)]

            history.append({
                'val_error_rate': float(val_error_rate),
                'train_loss': float(train_loss),
                'learning_rate': float(self.learning_rate.eval()),
                'difficulty': data_generator.difficulty()
            })

            with open('output/history.json', 'w') as outfile:
                json.dump(history, outfile)

            prev_error_rate = val_error_rate

    def predict(self, encoder_input_values, pad_symbol_idx=0):

        input_feed = {}
        for i, encoder_input in enumerate(self.encoder_inputs):
            input_feed[encoder_input.name] = encoder_input_values[:, i, :]

        for decoder_input in self.decoder_inputs:
            input_feed[decoder_input.name] = self.go_decoder_input_value

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

    def load(self, checkpoint_file):
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_file)

    def examples(self, data_generator):

        x, y = data_generator.next_batch(validation=True)

        # input_strings = decode_output_sequences(x, symbols=SYMBOLS)
        target_strings = decode_output_sequences(y, symbols=SYMBOLS)

        model_output = self.predict(x)

        pred_strings = decode_output_sequences(model_output, symbols=SYMBOLS)

        print(target_strings[:7])
        print(pred_strings[:7])
