#!/usr/bin/env python3

"""
perform a monte carlo tree search for a given board
with the help of a policy and value network
"""

import numpy as np
import random as rd
import math
from node import Node

WHITE = np.zeros((19, 19))
BLACK = np.ones((19, 19))

def update_board_player(board, player):
    """
    update third layer of board (white(0) => all zeros, black(1) => all ones)
    """
    board[2] = WHITE if not player else BLACK

def put_on_board(board, pos, player, value):
    """
    act on board, with player (0, 1), at x, y with value (0, 1)
    """
    board[player, pos[1], pos[0]] = value

def get_pos_on_board(board, nb_child):
    """
    take a number of children, and return x and y position on board 
    """
    # get unidimensional board of zero and one
    complete_board = np.add(board[0], board[1]).flatten()
    # get index of nb_child zero
    pos = (complete_board == 0).nonzero()[0][nb_child]
    # return x, y coordinates
    return pos % 19, pos // 19

def evaluate(maps, player, pos):
    """
    give a score to the current state
    """

    pmap = maps[player]

    line = pmap[pos[0]] # get line at pos
    p = pos[1]
    if '\x01\x01\x01\x01\x01' in ''.join(map(chr, line[max(p-4, 0):p+5])):
        return 1

    line = pmap[:,pos[1]] # get column at pos
    p = pos[0]
    if '\x01\x01\x01\x01\x01' in ''.join(map(chr, line[max(p-4, 0):p+5])):
        return 1

    line = pmap.diagonal(pos[1] - pos[0]) # get diagonal 1
    p = min(pos[0], pos[1])
    if '\x01\x01\x01\x01\x01' in ''.join(map(chr, line[max(p-4, 0):p+5])):
        return 1

    line = np.fliplr(pmap).diagonal(18 - pos[1] - pos[0]) # get diagonal 2
    p = min(pos[0], 18 - pos[1])
    if '\x01\x01\x01\x01\x01' in ''.join(map(chr, line[max(p-4, 0):p+5])):
        return 1

    # check if draw
    # for y in range(19):
    #     for x in range(19):
    #         if not maps[0, y, x] or not maps[1, y, x]:
    #             return ??
    return 0


def network(board):
    """
    tmp random function
    """
    return np.array([rd.randint(0, 20) for i in range(19 * 19)]), (rd.randint(0, 20) - 10) / 10

def expand(node, board, player):
    """
    evaluate state, and predict probabilities using neural network
    """

    # update board for network
    update_board_player(board, player)

    # run network
    p, v = network(board)

    # filter p to keep only possible moves
    p = p[np.where(np.add(board[0], board[1]).flatten() == 0)]

    # create children
    node.expand_children(p * (1 - 2 * player))

    # return v (turn to negative if black)
    return v * (1 - 2 * player)

def mc_search(node, board, player):
    """
    node: object Node
    board: np.array(3,19,19)
    deepness: integer
    player: 0 for white, 1 for black
    do actions on a level of deepness
    """

    # get array of score value, negative if black, and choose best move
    n = np.argmax(node.get_policy() * (1 - 2 * player))
    child = node.get_child(n)
    player ^= 1

    # update frequency
    child.add_frequency()

    # get coordinates of next move, and update board
    x, y = get_pos_on_board(board, n)
    put_on_board(board, (x, y), player, 1)

    # new node
    if child.leaf():
        value = evaluate(board, player, (x, y))
        # if not a terminating move
        if not value:
            value = expand(child, board, player)
    # keep searching
    else:
        value = mc_search(child, board, player)

    # clean board and back propagate
    put_on_board(board, (x, y), player, 0)
    child.score(value)
    return value

def turn(board, player, node):
    """
    board: np.array((3, 19, 19))
    take a state as input, and return board updated, policy vector, player and boolean for game status
    """

    # parameters: number of search
    trials = 1900

    # build tree
    for _ in range(trials):
        mc_search(node, board, player)

    # get move with hghest frequency
    n = node.get_max_frequency_move()

    # get coordinates of chosen move, and update board
    x, y = get_pos_on_board(board, n)
    put_on_board(board, (x, y), player, 1)

    # get final policy and child
    p = node.get_policy() * (1 - 2 * player)
    child = node.get_child(n)

    # return updated board, policy, and game status (0 or 1)
    return board, p, child, evaluate(board, player, (x, y))
