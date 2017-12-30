#!/usr/bin/env python3

"""
perform a monte carlo tree search for a given board
with the help of a policy and value network
"""

import numpy as np
import random as rd
import math
from node import Node

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

def get_max_children(board):
    """
    get number of possible actions: sum of non played tiles
    """
    complete_board = np.add(board[0], board[1])
    return np.sum(complete_board == 0) - 1

def mc_search(node, board, deepness, player):
    """
    node: object Node
    board: np.array(3,19,19)
    deepness: integer
    player: 0 for white, 1 for black
    do actions on a level of deepness
    """

    # update board player info (white(0) => 1, black(1) => 0)
    board[2].fill(player ^ 1)

    # random move
    nb_child = rd.randint(0, node.get_max_children())

    """
    eventually network policy head will give an array of probabilities
    nb_child wil be chosen as follow => 
    argmax ( for children c: Q ( score[c] ) + U ( prob[c] / 1 + frequency[c] ) )
    """

    # get child node and add frequency
    child = node.get_child(nb_child)
    child.add_frequency()

    # get coordinates of next move
    x, y = get_pos_on_board(board, nb_child)
    put_on_board(board, (x, y), player, 1)

    # get action value, negative if black
    value = evaluate(board, player, (x, y))
    value *= (1 - 2 * player)

    """
    here network value head will add its own value
    """

    if not value and deepness:
        # recursive call
        value += mc_search(child, board, deepness - 1, player ^ 1)

    # update score
    node.score(value)

    # clean board
    put_on_board(board, (x, y), player, 0)

    # return score
    return value

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


def turn(board):
    """
    board: np.array((3, 19, 19))
    take a state as input, and return board updated, policy vector, player and boolean for game status
    """

    # hyperparameters: number of search, deepness of the search
    trials = 1600
    deepness = 5

    # get root node
    node = Node(get_max_children(board))

    # get player turn
    player = 1 if board[2, 0, 0] == 0 else 0

    # build tree
    for _ in range(trials):
        mc_search(node, board, deepness - 1, player)

    # get array of score value, negative if black
    p = node.get_policy()
    p *= (1 - 2 * player)

    # get move with highest score value
    n = np.argmax(p)
    x, y = get_pos_on_board(board, n)

    # update board
    put_on_board(board, (x, y), player, 1)

    # get game status (0 or 1)
    e = evaluate(board, player, (x, y))

    return board, p, player, e
