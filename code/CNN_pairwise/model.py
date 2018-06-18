import tensorflow as tf
import numpy as np
import random

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term
def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output

class CNN():
    def __init__(self, hparams, mode):
        self.hparams = hparams
        self.mode = mode
        self.learning_rate = tf.Variable(float(hparams.learning_rate), trainable=False)
        self.clip_value = hparams.clip_value
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * hparams.decay_factor)
        self.seq_length1 = 40
        self.seq_length2 = 50
        self.vocab_size = hparams.from_vocab_size
        self.batch_size = hparams.batch_size
        self.emb_dim = hparams.emb_dim
        self.num_layers = hparams.num_layers
        self.num_units = hparams.num_units
        self.filter_sizes = hparams.filter_sizes
        self.num_filters = hparams.num_filters

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.input_x1 = tf.placeholder(tf.int32, [None, self.seq_length1])
            self.input_x2 = tf.placeholder(tf.int32, [None, self.seq_length2])
            self.input_x3 = tf.placeholder(tf.int32, [None, self.seq_length2])
            self.input_len1 = tf.placeholder(tf.int32, [None])
            self.input_len2 = tf.placeholder(tf.int32, [None])
            self.input_len3 = tf.placeholder(tf.int32, [None])
            self.input_y = tf.placeholder(tf.int32, [None])
            self.weight = tf.placeholder(tf.float32, [None])
            self.feature = tf.placeholder(tf.float32, [None])
        else:
            self.input_x1 = tf.placeholder(tf.int32, [None, self.seq_length1])
            self.input_x2 = tf.placeholder(tf.int32, [None, self.seq_length2])
            self.input_x3 = tf.placeholder(tf.int32, [None, self.seq_length2])
            self.input_len1 = tf.placeholder(tf.int32, [None])
            self.input_len2 = tf.placeholder(tf.int32, [None])
            self.input_len3 = tf.placeholder(tf.int32, [None])
            self.weight = tf.placeholder(tf.float32, [None])
            self.feature = tf.placeholder(tf.float32, [None])

        with tf.variable_scope("embedding") as scope:
            self.embeddings = tf.Variable(hparams.embeddings)
            # self.embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.emb_dim]), trainable=True)

        l2_loss = tf.constant(0.0)
        with tf.variable_scope("discriminator"):
            self.emb_inp1 = tf.nn.embedding_lookup(self.embeddings, self.input_x1)
            self.emb_inp1 = tf.expand_dims(self.emb_inp1, -1)

            self.emb_inp2 = tf.nn.embedding_lookup(self.embeddings, self.input_x2)
            self.emb_inp2 = tf.expand_dims(self.emb_inp2, -1)

            self.emb_inp3 = tf.nn.embedding_lookup(self.embeddings, self.input_x3)
            self.emb_inp3 = tf.expand_dims(self.emb_inp3, -1)


        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.input_keep_prob = self.hparams.input_keep_prob
            self.output_keep_prob = self.hparams.output_keep_prob
        else:
            self.input_keep_prob = 1.0
            self.output_keep_prob = 1.0
        # with tf.variable_scope("encoder"):
        #     encoder_emb_inp = tf.nn.embedding_lookup(self.embeddings, self.input_x)
        #     if self.num_layers > 1:
        #         encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([self._single_cell() for _ in range(self.num_layers)])
        #         encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([self._single_cell() for _ in range(self.num_layers)])
        #     else:
        #         encoder_cell_fw = self._single_cell()
        #         encoder_cell_bw = self._single_cell()
        #
        #     encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
        #         cell_fw=encoder_cell_fw,
        #         cell_bw=encoder_cell_bw,
        #         inputs=encoder_emb_inp,
        #         dtype=tf.float32,
        #         sequence_length=self.input_len)
        #     if self.num_layers > 1:
        #         encoder_state = []
        #         for layer_id in range(self.num_layers):
        #             fw_c, fw_h = bi_encoder_state[0][layer_id]
        #             bw_c, bw_h = bi_encoder_state[1][layer_id]
        #             # c = (fw_c + bw_c) / 2.0
        #             # h = (fw_h + bw_h) / 2.0
        #             c = tf.concat((fw_c, bw_c), axis=1)
        #             h = tf.concat((fw_h, bw_h), axis=1)
        #             state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)
        #             encoder_state.append(state)
        #             if layer_id == self.num_layers - 1:
        #                 encoder_state_vector = h
        #         encoder_state = tuple(encoder_state)
        #     else:
        #         fw_c, fw_h = bi_encoder_state[0]
        #         bw_c, bw_h = bi_encoder_state[1]
        #         # c = (fw_c + bw_c) / 2.0
        #         # h = (fw_h + bw_h) / 2.0
        #         c = tf.concat((fw_c, bw_c), axis=1)
        #         h = tf.concat((fw_h, bw_h), axis=1)
        #         encoder_state_vector = h
        #         encoder_state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)
        #
        #     self.hidden_state = encoder_state[self.num_layers - 1].h
        #     encoder_outputs_fw, encoder_outputs_bw = encoder_outputs
        #     memory = tf.concat([encoder_outputs_fw, encoder_outputs_bw], axis=2)
        #     self.memory = tf.expand_dims(memory, -1)

        pooled_outputs1 = []
        pooled_outputs2 = []
        pooled_outputs3 = []
        self.l2loss = tf.constant(0.0)
        for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.emb_dim, 1, num_filter]
                W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b1")
                self.l2loss += tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1)
                conv1 = tf.nn.conv2d(self.emb_inp1, W1, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
                h1 = tf.nn.tanh(tf.nn.bias_add(conv1, b1), name="relu1")
                pooled1 = tf.nn.max_pool(h1, ksize=[1, self.seq_length1 - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding="VALID", name="pool1")
                pooled_outputs1.append(pooled1)

                W2 = W1
                b2 = b1
                # W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
                # b2 = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b2")
                conv2 = tf.nn.conv2d(self.emb_inp2, W2, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
                h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2), name="relu2")
                pooled2 = tf.nn.max_pool(h2, ksize=[1, self.seq_length2 - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                         padding="VALID", name="pool2")
                pooled_outputs2.append(pooled2)

                conv3 = tf.nn.conv2d(self.emb_inp3, W2, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
                h3 = tf.nn.tanh(tf.nn.bias_add(conv3, b2), name="relu3")
                pooled3 = tf.nn.max_pool(h3, ksize=[1, self.seq_length2 - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                         padding="VALID", name="pool3")
                pooled_outputs3.append(pooled3)


        num_filters_total = sum(self.num_filters)
        self.h_pool1 = tf.concat(pooled_outputs1, 3)
        self.h1 = tf.reshape(self.h_pool1, [-1, num_filters_total])
        self.h_pool2 = tf.concat(pooled_outputs2, 3)
        self.h2 = tf.reshape(self.h_pool2, [-1, num_filters_total])
        self.h_pool3 = tf.concat(pooled_outputs3, 3)
        self.h3 = tf.reshape(self.h_pool3, [-1, num_filters_total])

        # self.h_pool_flat = tf.concat([self.h_pool_flat1, self.h_pool_flat2], axis=1)
        # with tf.name_scope("highway"):
        #     self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)
            # self.h_highway2 = highway(self.h_pool_flat2, self.h_pool_flat2.get_shape()[1], 1, 0)

            # Add dropout
        # with tf.name_scope("dropout"):
        #     if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        #         self.h_drop = tf.nn.dropout(self.h_highway, 0.5)
        #         # self.h_drop2 = tf.nn.dropout(self.h_highway2, 0.5)
        #     else:
        #         self.h_drop = tf.nn.dropout(self.h_highway, 0.5)
                # self.h_drop2 = tf.nn.dropout(self.h_highway2, 0.5)

        with tf.name_scope("output"):
            self.score1 = self.getCosine(self.h1, self.h2)
            self.score2 = self.getCosine(self.h1, self.h3)
            self.diff = tf.maximum(0.0, tf.subtract(0.05, tf.subtract(self.score1, self.score2)))
            self.loss = tf.reduce_sum(self.diff) + 0.01 * self.l2loss
            self.correct = tf.equal(0.0, self.diff)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")



        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.global_step = tf.Variable(0, trainable=False)
            with tf.name_scope("train_op"):
                optimizer = tf.train.AdamOptimizer(0.001)
                gradients, v = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
                self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                      global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables())

    def getCosine(self, q, a):
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            pooled_flat_1 = tf.nn.dropout(q, 0.75)
            pooled_flat_2 = tf.nn.dropout(a, 0.75)
        else:
            pooled_flat_1 = tf.nn.dropout(q, 1.0)
            pooled_flat_2 = tf.nn.dropout(a, 1.0)

        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_1), 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(pooled_flat_2, pooled_flat_2), 1))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_2), 1)
        score = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2), name="scores")
        return score

    def _single_cell(self, x=1):
        single_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units * x)
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell,
                                                    input_keep_prob=self.input_keep_prob,
                                                    output_keep_prob=self.output_keep_prob)
        return single_cell
    def get_batch(self, data, id=0, no_random=False):
        hparams = self.hparams
        xs1 = []
        xs2 = []
        xs3 = []
        ys = []
        x_len1 = []
        x_len2 = []
        x_len3 = []
        weight = []
        fs = []
        for i in range(self.batch_size):
            if not no_random:
                x1, x2, x3 = random.choice(data)
            else:
                x1, x2, x3 = data[id + i]
            pad_size = self.seq_length1  - len(x1)
            xs1.append(x1 +  [hparams.PAD_ID] * pad_size)
            x_len1.append(len(x1))

            pad_size = self.seq_length2 - len(x2)
            xs2.append(x2 + [hparams.PAD_ID] * pad_size)
            x_len2.append(len(x2))

            pad_size = self.seq_length2 - len(x3)
            xs3.append(x3 + [hparams.PAD_ID] * pad_size)
            x_len3.append(len(x3))


        return xs1, xs2, xs3, x_len1, x_len2, x_len3




    def train_step(self, sess, data):
        xs1, xs2, xs3, x_len1, x_len2, x_len3 = self.get_batch(data)
        feed = {
            self.input_x1: xs1,
            self.input_len1: x_len1,
            self.input_x2: xs2,
            self.input_len2: x_len2,
            self.input_x3: xs3,
            self.input_len3: x_len3
        }
        loss, accuracy, global_step, _ = sess.run([self.loss,
                                                   self.accuracy,
                                                   self.global_step,
                                                   self.train_op], feed_dict=feed)
        return loss, global_step, accuracy

    def eval_step(self, sess, data, id=0, no_random=True):
        xs1, xs2, xs3, x_len1, x_len2, x_len3 = self.get_batch(data, id, no_random)
        feed = {
            self.input_x1: xs1,
            self.input_len1: x_len1,
            self.input_x2: xs2,
            self.input_len2: x_len2,
            self.input_x3: xs3,
            self.input_len3: x_len3
        }
        loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed)
        return loss, accuracy

    def infer_step(self, sess, data, id=0, no_random=True):
        xs1, xs2, xs3, x_len1, x_len2, x_len3 = self.get_batch(data, id, no_random)
        feed = {
            self.input_x1: xs1,
            self.input_len1: x_len1,
            self.input_x2: xs2,
            self.input_len2: x_len2,
            self.input_x3: xs3,
            self.input_len3: x_len3
        }
        predictions, accuracy = sess.run([self.score2, self.accuracy], feed_dict=feed)
        return predictions, accuracy

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def lr_decay(self, sess):
        return sess.run(self.learning_rate_decay_op)




