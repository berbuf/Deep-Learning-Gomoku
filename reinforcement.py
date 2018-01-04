 #!/usr/bin/env python3

"""
play a game against oneself until the end
save labels (state, policy, winning) to file label_{num_game}.label
"""

import numpy as np
from mc_tree_search import turn, expand
from node import Node
from unit_tests import conv_map
import os

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

def print_board(board):
    """
    debug, print current map to term with colours
    """
    board = (board[:,:,0] + board[:,:,1] * 2)
    print ("board:")
    for line in board:
        l = ""
        for tile in line:
            if tile == 1:
                l += "\033[34;10m" + str(tile) + "\033[0m "
            elif tile == 2:
                l += "\033[31;10m" + str(tile) + "\033[0m "
            else:
                l += "  "
        print (l)

def start_state():
    return conv_map(np.array([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

def init_game(network_1, network_2):
    """
    init game board, first node, next player turn
    """
    # start with blank map
    #board = np.zeros((19, 19, 3), np.int8)
    # init board with given state
    board = start_state()
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
    pos, board, p, p_node, status = turn(board, player, p_node, p_net)
    # update opponent node with player choice
    o_node = update_turn(board, player ^ 1, o_node, o_net, pos)
    if labels is not None:
        save_tmp_label(labels, board, p, player)

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
            return

        status, p_2, p_1 = sequence(board, 1, p_2, net_2, p_1, net_1, None)
        if status:
            save_final_label(labels, filename, 1)
            return
