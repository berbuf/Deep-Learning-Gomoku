#!/usr/bin/env python3

"""
play a game against oneself until the end
save labels (state, policy, winning) to file label_{num_game}.label
train network from batches of labels
"""

import os
import numpy as np
import random as rd
from mcts import mcts, expand
from node import Node
from network import Network
from utils_board import init_map, print_board

def save_tmp_label(turns, board, p, player):
    """
    append board, policy vector and current player to tmp file
    shape: ( (19, 19, 3), (361), (1) )
    """
    turns.append((board, p, player))

def save_final_label(turns, winner, filename):
    """
    rewrite tmp file and change player to 1 if == winner, else -1
    """
    turns = list(map(lambda x: (x[0], x[1], 1 if x[2] == winner else -1), turns))
    if os.path.exists(filename):
        turns += list(np.load(filename))
    np.save(filename, np.array(turns))

def init_game(network_1, network_2):
    """
    init game board, first node, next player turn
    """
    #board = init_map()
    board = np.zeros((19, 19, 3), np.int8)
    # player 1
    node_p_1 = Node(0)
    expand(node_p_1, board, 0, network_1)
    # player 2
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
    pos, board, p, p_node, status = mcts(board, player, p_node, p_net)
    o_node = update_turn(board, player ^ 1, o_node, o_net, pos)
    save_tmp_label(labels, board, p, player)
    #print_board(board)
    return status, p_node, o_node

def game(net_1, net_2):
    """
    take identifier of a game and play it until the end
    num_game: integer
    """
    board, p_1, p_2 = init_game(net_1, net_2)
    labels = []
    while (True):
        status, p_1, p_2 = sequence(board, 0, p_1, net_1, p_2, net_2, labels)
        if status:
            print_board(board)
            return labels, 0
        status, p_2, p_1 = sequence(board, 1, p_2, net_2, p_1, net_1, labels)
        if status:
            print_board(board)
            return labels, 1

def random_rotation(s, p, z):
    """
    apply random rotation on label
    """
    r = rd.randint(0, 3)
    s[:,:,0] = np.rot90(s[:,:,0], r, (0, 1))
    s[:,:,1] = np.rot90(s[:,:,1], r, (0, 1))
    p = np.rot90(p.reshape((19, 19)), r, (0, 1)).flatten()
    return (s, p, z)

def evaluation(number_games, champion, trainee):
    """
    take number_games, version and trainee
    return percentage of victory after n games
    """
    win = 0
    i = 0
    while (i < number_games):
        print(i, end=" ", flush=True)
        _, winner = game(champion, trainee)
        win += winner

        i += 1

        print(i, end=" ", flush=True)
        _, winner = game(trainee, champion)
        win += not winner
        i += 1

    return win / number_games * 100

def training(number_training, batch_size, size_train_labels, version, trainee):
    """
    take number_training, version and trainee
    train on label
    """
    labels = list(np.load("../labels/labels_" + str(version) + ".npy"))[-size_train_labels:]
    for i in range(number_training):
        print(i, end=" ", flush=True)

        # random batch
        batch = rd.sample(labels, min(batch_size, len(labels)))

        # random transformation
        batch = [ random_rotation(s, p, z) for s, p, z in batch ]

        # training
        batch = np.array(batch)
        trainee.train(batch[:,0], batch[:,1], batch[:,2])

    del labels

def self_play(number_games, player, version):
    """
    take number_games version, and produce labels
    """
    path_label = "../labels/labels_" + str(version) + ".npy"
    for i in range(number_games):
        print(i, end=" ", flush=True)
        tmp_labels, winner = game(player, player)
        save_final_label(tmp_labels, winner, path_label)

def reinforcement():
    """
    train model against itself
    """
    number_games = 10
    number_training = 10
    number_evaluation = 6
    batch_size = 1024
    size_train_labels = 50000
    epoch = 0

    # get champion version
    version = 1
    trainee = Network(version)
    champion = Network(version)
    trainee.save_session()

    while (True):
        # produce labels from best version
        print ("self play, number_games:", number_games, "version:", version)
        self_play(number_games, champion, version)

        # cloned trainee learns from labels
        print ("\ntraining, number_training:", number_training, "version:", version)
        training(number_training, batch_size, size_train_labels, version, trainee)

        # if trainee beats champion, trainee becomes champion, new trainee is cloned from it
        print ("\nevaluation, number_evaluation:", number_evaluation)
        score = evaluation(number_evaluation, champion, trainee)
        if score > 55:
            os.remove("../labels/labels_" + str(version) + ".npy")
            version += 1
            trainee.save_session()
            champion = Network(version)
            trainee = Network(version)

        epoch += 1
        print ("\nfinal evaluation score:", score, "actual version", version, " number of epoch", epoch)

    trainee.save_session()

if __name__ == '__main__':
    # training
    reinforcement()
