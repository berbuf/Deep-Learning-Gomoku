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
    complete_board = np.add(board[0], board[1]).flatten()
    # get index of nb_child zero
    pos = (complete_board == 0).nonzero()[0][nb_child]
    # return x, y coordinates
    return pos % 19, math.ceil(pos / 19)

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

def evaluate(board):
    """
    give a score to the current state
    """
    return 0

def policy():
    """
    establish score of all possible action
    """

def get_max_children(board):
    """
    get number of possible actions: sum of non played tiles
    """
    complete_board = np.add(board[0], board[1])
    return np.sum(complete_board == 0)

def turn(board):
    """
    take a state as input, and return a position 
    """

    # get root node
    node = Node(get_max_children(board))

    # build tree
    node = mc_search(node, board, 3)

    return 0, 0

if __name__ == '__main__':

    # create board
    board = np.zeros((3, 19, 19), np.float32)

    # get best move
    pos = turn(board)
