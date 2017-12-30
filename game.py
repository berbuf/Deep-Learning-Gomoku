 #!/usr/bin/env python3

"""
play a game until the end
save labels (state, policy, winning) to file label_{num_game}.label
"""

import numpy as np
from mc_tree_search import turn

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
    board = np.add(board[0], board[1] * 2)
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

def game(num_game):
    """
    take identifier of a game and play it until the end
    num_game: integer
    """

    # init game board
    board = np.zeros((3, 19, 19), np.int8)
    player = 1

    e = 0
    while (not e):
        # next player
        player ^= 1

        # run mcts simulation (p: array of scores, e: game has finished )
        board, p, e = turn(board, player)

        # save board state, policy vector and current player in tmp folder
        save_tmp_label(board, p, player)

        # debug
        print_board(board)

    print("end game", player + 1)

    # rewrite tmp file to num_game file with final winner info
    save_final_label(num_game, player)
