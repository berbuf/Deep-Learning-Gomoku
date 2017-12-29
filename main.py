#!/usr/bin/env python3

import numpy as np
from mc_tree_search import turn


# enironment of play
# 2 ai playing against each other





if __name__ == '__main__':
    board = np.zeros((3, 19, 19), np.int8)

    # get best move
    pos = turn(board)


    # jouer
    # 2 players
