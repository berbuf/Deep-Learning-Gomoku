#!/usr/bin/env python3
"""
misc functions about board management
"""

import numpy as np

def print_policy(board, p):
    """
    debug: print policy
    """
    board = (board[:,:,0] + board[:,:,1] * 2)
    for y in range(19):
        l = ""
        for x in range(19):
            tile = p[y * 19 + x]
            if tile < -0.7:
                c = "31" # red
            elif tile < -0.3:
                c = "33" # yellow
            elif tile < 0:
                c = "35" # magenta
            elif tile == 0:
                c = "37" # white
            elif tile < 0.3:
                c = "34" # blue
            elif tile < 0.7:
                c = "36" # cyan
            else:
                c = "32" # green
            if board[y, x] != 0:
                l += "  X   "
            else:
                l += "\033[" + c + ";10m" + "{0:.2f}".format(tile) + "\033[0m  "
        print (l)

def print_board(board):
    """
    debug, print current map to term with colours
    """
    board = (board[:,:,0] + board[:,:,1] * 2)
    print ("board:")
    l = "  "
    for i in range(19):
        l += "{0:02d} ".format(i)
    print (l)
    for i, line in enumerate(board):
        l = "{0:02d} ".format(i)
        for tile in line:
            if tile == 1:
                l += "\033[34;10m" + str(tile) + "\033[0m  "
            elif tile == 2:
                l += "\033[31;10m" + str(tile) + "\033[0m  "
            else:
                l += "   "
        print (l)

def conv_map(m):
    """
    convert test map to shape (19, 19, 3)
    """
    p1 = np.copy(m)
    np.place(p1, p1 == 2, 0)
    p2 = np.copy(m)
    np.place(p2, p2 == 1, 0)
    np.place(p2, p2 == 2, 1)
    c = np.zeros(19) if np.sum( p1 ) == np.sum( p2 ) else np.ones(19)
    return np.array( [ list(zip(la, lb, c)) for la, lb in zip(p1, p2) ], np.int8 )

def init_map():
    """
    easy to configure state
    """
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

