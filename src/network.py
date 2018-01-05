import tensorflow as tf
import numpy as np
from tqdm import tqdm

"""
class Network
"""

class Network(object):
    def __init__(self, version):
        """
        init network
        """
        self._state = tf.placeholder(tf.float32, shape=[None, 19, 19, 3])
        self._p_head, self._v_head = network(self._state)
        self._loss, self._train_p_mcts, self._train_winner = loss_function(self._state,
                                                                           self._p_head,
                                                                           self._v_head)
        self._glob = tf.global_variables_initializer()
        self._sess = tf.Session()
        self._sess.run(self._glob)

    def infer(self, board):
        """
        infer policy and value from board state
        """
        return self._sess.run([self._p_head, self._v_head],
                              feed_dict={self._state: board[None, :]})

    def train(self, board, p, z):
        """
        train sequence
        save trained model
        board: current board
        p: probability computed by mcts for board
        z: winner of the game
        """
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
    Implementation of a convolutional layer with 256 filter (3x3),
    batch normalization and rectifier non linearity (reLU)
    """

    conv = convolution(input=input, filters=256, ksize=3)
    bn = tf.layers.batch_normalization(conv)
    relu = tf.nn.relu(bn)
    return relu

def res_layer(input):
    """
    Implementation of a residual layer with 256 filter (3x3)
    """

    conv = convolution(input=input, filters=256, ksize=3)
    bn = tf.layers.batch_normalization(conv)
    relu = tf.nn.relu(bn)
    conv = convolution(input=relu, filters=256, ksize=3)
    bn = tf.layers.batch_normalization(conv)
    skip = tf.add(bn, input)
    relu = tf.nn.relu(skip)
    return relu

def value_head(input):
    """
    The value head
    """
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
    conv = convolution(input=input, filters=2, ksize=1)
    bn = tf.layers.batch_normalization(conv)
    relu = tf.nn.relu(bn)
    flatten = tf.reshape(relu, [-1, 19 * 19 * 2])
    fc = fully_connected(flatten, units=19 * 19)
    return fc

def network(input):
    layer = conv_layer(input)
    for _ in range(20):
        layer = res_layer(layer)
    policy = policy_head(layer)
    value = value_head(layer)
    return policy, value

def loss_function(state, p_head, v_head):
    
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
