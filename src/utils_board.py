#!/usr/bin/env python3                                                                                   

"""                                                                                                      
misc functions about board management
"""

import numpy as np

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

