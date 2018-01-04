#!/usr/bin/env python3

"""
play a game against oneself until the end
save labels (state, policy, winning) to file label_{num_game}.label
train network from batches of labels
"""

import os
import numpy as np
from mcts import mcts, expand
from node import Node
from network import Network
from utils_board import init_map, print_board

def save_tmp_label(turns, board, p, player):
    """
    append board, policy vector and current player to tmp file
    """
    # shape: ( (19, 19, 3), (361), (1) )
    turns.append((board, p, player))

def save_final_label(turns, filename, winner):
    """
    rewrite tmp file and change player to 1 if == winner, else -1
    format => np.array => ((19, 19, 3), 19 * 19, 1) => 19, 19, 5
    """
    # change z value
    turns = list(map(lambda x: (x[0], x[1], 1 if x[2] == winner else -1), turns))
    # append to existing array
    if os.path.exists(filename):
        turns += list(np.load(filename))
    # save to disk
    np.save(filename, np.array(turns))

def load_nparray(filename):
    """
    load labels from file
    """
    if os.path.exists(filename):
        array = np.load(filename)
        os.remove(filename)
        return array

def train_from_file(filename, network):
    """
    train network once
    """
    if os.path.exists(filename):
        array = np.load(filename)
        network.train(array['boards'], array['p'], array['z'])
        os.remove(filename)

def init_game(network_1, network_2):
    """
    init game board, first node, next player turn
    """
    # start with blank map
    # board = np.zeros((19, 19, 3), np.int8)
    # or init board with given state
    board = init_map()
    # init player 1 tree
    node_p_1 = Node(0)
    expand(node_p_1, board, 0, network_1)
    # init player 2 tree
    node_p_2 = Node(0)
    expand(node_p_2, board, 1, network_2)
    return board, node_p_1, node_p_2

def update_turn(board, player, node, network, pos):
    """
    take board, player number, current node, player network and opponent move
    return child node according to opponent move
    """
    # get child from number of empty moves before opponent move
    complete_board = (board[:,:,0] + board[:,:,1]).flatten()
    child = node.get_child(np.sum((complete_board[:(pos[1] * 19 + pos[0])] == 0)))
    # if unexplored yet
    if child.leaf():
        expand(child, board, player, network)
    return child

def sequence(board, player, p_node, p_net, o_node, o_net, labels):
    """
    perform a complete turn
    take state info, player objects, and save boolean
    return game status (O, 1), updated current and opponent nodes
    """
    # run mcts simulation: return pos move, new board, policy p, new node, game status
    pos, board, p, p_node, status = mcts(board, player, p_node, p_net)
    # update opponent node with player choice
    o_node = update_turn(board, player ^ 1, o_node, o_net, pos)
    if labels is not None:
        save_tmp_label(labels, board, p, player)
    # debug
    print_board(board)
    return status, p_node, o_node

def game(net_1, net_2, filename):
    """
    take identifier of a game and play it until the end
    num_game: integer
    """
    board, p_1, p_2 = init_game(net_1, net_2)
    labels = []

    while (True):
        status, p_1, p_2 = sequence(board, 0, p_1, net_1, p_2, net_2, labels)
        if status:
            save_final_label(labels, filename, 0)
            break

        status, p_2, p_1 = sequence(board, 1, p_2, net_2, p_1, net_1, None)
        if status:
            save_final_label(labels, filename, 1)
            break

def reinforcement():
    """
    train model against itself
    """
    # parameters
    version = "1.0"
    path_label = "../labels/labels_" + version + ".npy"
    number_of_games = 1

    player_1 = Network(-1)
    player_2 = Network(-1)

    for num_game in range(number_of_games):

        # play a game until the end
        game(player_1, player_2, path_label)

        """
        #trainning all 3 games (for example)
        if num_game % 1 == 0:
            # train network
            train_from_file(path_label, player_2)
        """
