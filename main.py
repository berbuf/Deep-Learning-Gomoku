#!/usr/bin/env python3

import numpy as np
from mc_tree_search import turn

if __name__ == '__main__':
    board = np.zeros((3, 19, 19))

    # get best move
    pos = turn(board)
