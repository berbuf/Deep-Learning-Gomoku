import os.path
import tensorflow as tf
import numpy as np

"""
class Network
"""

class Network(object):
    def __init__(self, version):
        """
        init network
        """
        path_to_restore_model = "../models/model-" + str(version) + ".ckpt";
        self.version = version

        if os.path.isfile(path_to_restore_model):
            print("restore")
            saver = tf.train.Saver()
            saver.restore(self._sess, path_to_restore_model)
        else:
            self._graph = tf.Graph()
            with self._graph.as_default():
                self._state = tf.placeholder(tf.float32, shape=[None, 19, 19, 3])
                self._p_head, self._v_head = network(self._state)
                self._loss, self._train_p_mcts, self._train_winner = loss_function(self._state,
                                                                                   self._p_head,
                                                                                   self._v_head)
                self._glob = tf.global_variables_initializer()
                self._sess = tf.Session(graph=self._graph)
                self._sess.run(self._glob)
                writer = tf.summary.FileWriter('./logs')
                writer.add_graph(self._graph)

    def infer(self, board):
        """
        infer policy and value from board state
        """
        return self._sess.run([self._p_head, self._v_head],
                              feed_dict={self._state: board[None, :]})

    def save_session(self):
        with self._graph.as_default():
            saver = tf.train.Saver()
            saver_path = saver.save(self._sess, "../models/model-" + str(self.version) + ".ckpt")
            print("model saved in %s" %saver_path)
        return

    def train(self, board, p, z):
        """
        train sequence
        save trained model
        board: current board
        p: probability computed by mcts for board
        z: winner of the game
        """
        board = np.array(board.tolist())
        p = np.array(p.tolist())
        z = np.array(z.tolist())
        lr = 0.5
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self._loss)
        self._sess.run([optimizer],
                       feed_dict={self._state: board,
                                  self._train_p_mcts: p,
                                  self._train_winner: z})

def convolution(input, filters, ksize):
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    conv = tf.layers.conv2d(
        inputs=input,
        filters=filters,
        kernel_size=[ksize, ksize],
        padding='SAME',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=True,
        bias_initializer=initializer,
        bias_regularizer=regularizer,
        activation=tf.nn.relu)
    return conv

def fully_connected(flatten_input, units):
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)

    fc = tf.layers.dense(
        inputs=flatten_input,
        units=units,
        activation=None,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        use_bias=True,
        bias_initializer=initializer,
        bias_regularizer=regularizer)
    return fc

def conv_layer(input):
    """
    Implementation of a convolutional layer with 64 filter (3x3),
    batch normalization and rectifier non linearity (reLU)
    """

    with tf.variable_scope('conv'):
        conv = convolution(input=input, filters=64, ksize=3)
        bn = tf.layers.batch_normalization(conv)
        relu = tf.nn.relu(bn)
        return relu

def res_layer(input, id):
    """
    Implementation of a residual layer with 64 filter (3x3)
    """
    with tf.variable_scope('res_' + str(id) ) as scope:
        conv = convolution(input=input, filters=64, ksize=3)
        bn = tf.layers.batch_normalization(conv)
        relu = tf.nn.relu(bn)
        conv = convolution(input=relu, filters=64, ksize=3)
        bn = tf.layers.batch_normalization(conv)
        skip = tf.add(bn, input)
        relu = tf.nn.relu(skip)
        return relu

def value_head(input):
    """
    The value head
    """

    with tf.variable_scope('value_head'):
        conv = convolution(input=input, filters=1, ksize=1)
        bn = tf.layers.batch_normalization(conv)
        relu = tf.nn.relu(bn)
        flatten = tf.reshape(relu, [-1, 19 * 19 * 1])
        fc = fully_connected(flatten, units=256)
        relu = tf.nn.relu(fc)
        fc = fully_connected(relu, units=1)
        tanh = relu = tf.nn.tanh(fc)
        output = tanh[:, 0]
        return output

def policy_head(input):
    """
    The policy head
    """
    with tf.variable_scope('policy_head'):
        conv = convolution(input=input, filters=2, ksize=1)
        bn = tf.layers.batch_normalization(conv)
        relu = tf.nn.relu(bn)
        flatten = tf.reshape(relu, [-1, 19 * 19 * 2])
        fc = fully_connected(flatten, units=19 * 19)
        tanh = tf.nn.tanh(fc)
        return tanh

def network(input):
    print('network')
    layer = conv_layer(input)
    for i in range(16):
        layer = res_layer(layer, i)
    policy = policy_head(layer)
    value = value_head(layer)
    return policy, value

def loss_function(state, p_head, v_head):

    with tf.variable_scope('loss'):
        c = 0.01
        p = tf.placeholder(tf.float32, [None, 19 * 19])
        z = tf.placeholder(tf.float32, [None])

        mean_square = tf.reduce_mean(tf.squared_difference(v_head, z))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(p * tf.log(p_head), reduction_indices=[1]))

        regularizer = tf.nn.l2_loss(state)
        loss = mean_square + cross_entropy + c * regularizer

        return loss, p, z

"""
if __name__ == '__main__':

    state = tf.constant(.0, shape=[1, 19, 19, 3])
    policy, value = network(state)

    print ("config graph")
    graph_location = './logs'
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(graph_location)

    with tf.Session() as sess:
        writer.add_graph(sess.graph)
"""
