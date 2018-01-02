 #!/usr/bin/env python3

"""
play a game against oneself until the end
save labels (state, policy, winning) to file label_{num_game}.label
"""

import numpy as np
from mc_tree_search import turn, expand
from node import Node

def save_tmp_label(board, p, player):
    """
    append board, policy vector and current player to tmp file
    format => np.array => ((3, 19, 19), 19 * 19, 1)
    """
    return 

def save_final_label(num, winner):
    """
    rewrite tmp file and change player to 1 if == winner, else -1
    format => np.array => ((3, 19, 19), 19 * 19, 1)
    """
    return 

def print_board(board):
    """
    debug, print current map to term with colours
    """
    board = (board[:,:,:,0] + board[:,:,:,1] * 2)[0]
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

def init_game(network_1, network_2):
    """
    init game board, first node, next player turn
    """
    # init board
    board = np.zeros((1, 19, 19, 3), np.int8)
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
    complete_board = (board[:,:,:,0] + board[:,:,:,1]).flatten()
    child = node.get_child(np.sum((complete_board[:(pos[1] * 19 + pos[0])] == 0)))
    # if unexplored yet
    if child.leaf():
        expand(child, board, player, network)
    return child

def turn_sequence(board, player, node_current, net_current, node_opponent, net_opponent, save):
    """
    perform a complete turn
    take state info, player objects, and save boolean
    return game status (O, 1), updated cuurent and opponent nodes
    """
    # run mcts simulation: return pos move, new board, policy p, new node, game status
    pos, board, p, node_current, status = turn(board, player, node_current, net_current)
    # update opponent node with player choice
    node_opponent = update_turn(board, player ^ 1, node_opponent, net_opponent, pos)
    if save:
        save_tmp_label(board, p, player)
    return status, node_current, node_opponent

def game(network_1, network_2, num_game):
    """
    take identifier of a game and play it until the end
    num_game: integer
    """
    board, node_p_1, node_p_2 = init_game(network_1, network_2)
    
    winner = 0
    while (not winner):

        status, node_p_1, node_p_2 = turn_sequence(board, 0, node_p_1, network_1, node_p_2, network_2, True)
        if status:
            winner = 1
            break
        print_board(board)

        status, node_p_2, node_p_1 = turn_sequence(board, 1, node_p_2, network_2, node_p_1, network_1, False)
        if status:
            winner = 2
        print_board(board)

    print("end game", winner - 1)

    # save board state, policy vector and current player in tmp folder
    save_final_label(num_game, winner - 1)
