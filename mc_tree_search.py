#!/usr/bin/env python3

"""
perform a monte carlo tree search for a given board
with the help of a policy and value network
"""

import numpy as np
import random as rd
import math
from node import Node

def update_board_player(board, player):
    """
    update third layer of board (white(0) => all zeros, black(1) => all ones)
    """
    board[:,:,2] = player

def put_on_board(board, pos, player, value):
    """
    act on board, with player (0, 1), at x, y with value (0, 1)
    """
    board[pos[1], pos[0], player] = value

def get_pos_on_board(board, nb_child):
    """
    take a number of children, and return x and y position on board 
    """
    # get unidimensional board of zero and one
    complete_board = (board[:,:,0] + board[:,:,1]).flatten()
    # get index of nb_child zero
    pos = (complete_board == 0).nonzero()[0][nb_child]
    return pos % 19, pos // 19

def evaluate(board, player, pos):
    """
    give a score to the current state
    board => (19, 19, 3)
    player => 0 or 1
    pos => (x, y)
    """
    maps = [ board[:,:,0], board[:,:,1] ]
    pmap = maps[player]

    line = pmap[pos[1]] # get line at pos
    p = pos[0]
    if '\x01\x01\x01\x01\x01' in ''.join(map(chr, line[max(p-4, 0):p+5])):
        return 1

    line = pmap[:,pos[0]] # get column at pos
    p = pos[1]
    if '\x01\x01\x01\x01\x01' in ''.join(map(chr, line[max(p-4, 0):p+5])):
        return 1

    line = pmap.diagonal(pos[0] - pos[1]) # get diagonal 1
    p = min(pos[0], pos[1])
    if '\x01\x01\x01\x01\x01' in ''.join(map(chr, line[max(p-4, 0):p+5])):
        return 1

    line = np.fliplr(pmap).diagonal(18 - pos[0] - pos[1]) # get diagonal 2
    p = min(pos[1], 18 - pos[0])
    if '\x01\x01\x01\x01\x01' in ''.join(map(chr, line[max(p-4, 0):p+5])):
        return 1

    # check if draw
    # for y in range(19):
    #     for x in range(19):
    #         if not maps[0, y, x] or not maps[1, y, x]:
    #             return ??
    return 0

def tmpnetwork(board):
    """
    tmp random function
    """

    return (np.array( [ rd.randint(0, 10) / 10 for i in range(19 * 19) ] ), 0)

def expand(node, board, player, network):
    """
    expand node
    predict state value, and probability for children using neural network
    """
    update_board_player(board, player)
    # fast, random network
    """
    p, v = tmpnetwork(board)
    p = p[np.where((board[:,:,0] + board[:,:,1]).flatten() == 0)] 
    # run network, (negative if black)
    """
    p, v = network.infer(board)
    p = p[0][np.where((board[:,:,0] + board[:,:,1]).flatten() == 0)] 

    p *= (1 - 2 * player)
    v *= (1 - 2 * player)
    node.expand_children(p)
    return v

def select(node, board, player):
    """
    return chosen node, updated board, new coordinates
    """
    # choose next node
    n = np.argmax(node.get_policy() * (1 - 2 * player))
    #print (node.get_policy())
    child = node.get_child(n)
    child.add_frequency()
    # get coordinates of next move, and update board
    x, y = get_pos_on_board(board, n)
    put_on_board(board, (x, y), player, 1)
#    if player == 1 and x == 1 and y == 0:
#        print (node.get_policy())
#    if player == 0:
#        print (node.get_policy())
#    if player == 1:
#        print (node.get_policy())
    return child, board, x, y, player ^ 1

def mc_search(node, board, player, network):
    """
    node: object Node
    board: np.array(3,19,19)
    player: 0 for white, 1 for black
    do actions on a level of deepness
    """
    child, board, x, y, next_player = select(node, board, player)
    # evaluate or keep searching
    if child.leaf():
        value = evaluate(board, player, (x, y))
        # if not a winning move
        if not value:
            value = expand(child, board, next_player, network)
    else:
        value = mc_search(child, board, next_player, network)
    # clean board and back propagate
    put_on_board(board, (x, y), player, 0)
    child.score(value)
    return value

def turn(board, player, root, network):
    """
    board: np.array((3, 19, 19))
    take board, player turn (0, 1), root node
    return next move, updated board, policy vector, next root and boolean for game status
    """
    # parameters: number of search
    trials = 5

    # build tree
    for i in range(trials):
        mc_search(root, board, player, network)

    n = root.get_max_frequency_move()

    # get coordinates of chosen move, and update board
    x, y = get_pos_on_board(board, n)
    put_on_board(board, (x, y), player, 1)

    return ((x, y), board, root.get_policy() * (1 - 2 * player),
            root.get_child(n), evaluate(board, player, (x, y)))
