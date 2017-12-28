#!/usr/bin/env python3

import numpy as np
import random as rd
import math
from node import Node

def put_on_board(board, x, y, value):
    """
    act on board, with player (0, 1), at x, y with value (0, 1)
    """
    player = math.ceil(board[0][0][0])
    board[player][y][x] = value

def get_pos_on_board(board, nb_child):
    """
    take a number of children, and return x and y position on board 
    """
    # get unidimensional board of zero and one
    complete_board = np.add(board[1], board[2]).flatten()
    # get index of nb_child zero
    pos = (complete_board == 0).nonzero()[0][nb_child]
    # return x, y coordinates
    return pos % 19, pos // 19

def is_leaf(node, board, deepness):
    """
    check if the tree has reached a leaf
    """
    # check scoring
    e = evaluate(board)
    if e != 0:
        node.score(e)
        return True
    # check deepness
    if deepness < 0:
        return True
    return False

def mc_search(node, board, deepness):
    """
    do actions on a level of deepness
    """

    # add frequency, and check leaf
    node.add_frequency()
    if is_leaf(node, board, deepness):
        return node

    # random move
    nb_child = rd.randint(0, node.get_max_children())

    # get child node
    child = node.get_child(nb_child)

    # get coordinates of next move
    x, y = get_pos_on_board(board, nb_child)

    # put on board
    put_on_board(board, x, y, 1)

    # recursive call
    mc_search(child, board, deepness - 1)

    # clean board
    put_on_board(board, x, y, 0)

    return node

def evaluate(maps, pos):
    """
    give a score to the current state
    """

    player = maps[2, 0, 0]
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


def policy(node, board):
    """
    establish score of all possible action
    """

    p = node.get_policy()
    print ("policy:")
    print (p[:100])

    n = np.argmax(p)
    x, y = get_pos_on_board(board, n)
    print ("best move:")
    print ("x : ", x, " y: ", y, " prob: ", p[n])

    return x, y

def get_max_children(board):
    """
    get number of possible actions: sum of non played tiles
    """
    complete_board = np.add(board[1], board[2])
    return np.sum(complete_board == 0) - 1

def turn(board):
    """
    take a state as input, and return a position 
    """

    # get root node
    node = Node(get_max_children(board))

    # build tree
    for _ in range(10000):
        node = mc_search(node, board, 3)
        
    # get best move
    x, y = policy(node, board)

    return x, y
