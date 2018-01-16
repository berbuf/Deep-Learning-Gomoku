#!/usr/bin/env python3

import numpy as np
from mcts import evaluate, expand, mcts
from network import Network
from node import Node
from utils_board import conv_map

def test_evaluate_graphic():
    """
    test evaluate with graphical output
    """
    while input('press q to quit: ') != 'q':
        tmp = np.random.randint(3, size=(19, 19))
        board = np.zeros((1, 19, 19, 3), dtype=int)
        for y in range(19):
            for x in range(19):
                if tmp[y, x] == 1:
                    board[0, y, x, 0] = 1
                elif tmp[y, x] == 2:
                    board[0, y, x, 1] = 1

        wpos = []
        for y in range(19):
            for x in range(19):
                if evaluate(board, 0, [y, x]):
                    wpos.append([y, x])
        for y in range(19):
            for x in range(19):
                if evaluate(board, 1, [y, x]):
                    wpos.append([y, x])

        for y in range(19):
            for x in range(19):
                v = board[0, y, x, 0] + board[0, y, x, 1] * 2
                if [y, x] in wpos:
                    print('\033[92m{}\033[0m'.format(v), end='')
                else:
                    print(v, end='')
            print()
    pass

def test_winning_move_when_multiple():
    """
    test if network find winning move among many
    return true if winning move is found
    return else otherwise
    """
    while True:
        has_winning_move = 0
        tmp = np.random.randint(3, size=(19, 19))
        board = np.zeros((1, 19, 19, 3), dtype=int)

        for y in range(19):
            for x in range(19):
                if tmp[y, x] == 1:
                    board[0, y, x, 0] = 1
                    if evaluate(board, 0, [y, x]):
                        board[0, y, x, 0] = 0
                        has_winning_move = 1
                elif tmp[y, x] == 2:
                    board[0, y, x, 1] = 1
                    if evaluate(board, 1, [y, x]):
                        board[0, y, x, 1] = 0
                        has_winning_move = 2
        if not has_winning_move:
            continue

        network = Network(0)
        node = Node(0)
        expand(node, board, has_winning_move - 1, network)

        mcts(board, has_winning_move - 1, node, network)

        for y in range(19):
            for x in range(19):
                if evaluate(board, has_winning_move - 1, [y, x]):
                    return True

        return False

def test_winning_move_when_one():
    """
    test if network find winning move among one
    return true if winning move is found
    return else otherwise
    """
    while True:
        has_winning_move = 0
        tmp = np.random.randint(3, size=(19, 19))
        board = np.zeros((1, 19, 19, 3), dtype=int)

        for y in range(19):
            for x in range(19):
                if tmp[y, x] == 1:
                    board[0, y, x, 0] = 1
                    if evaluate(board, 0, [y, x]):
                        board[0, y, x, 0] = 0
                        if has_winning_move:
                            has_winning_move = 1
                elif tmp[y, x] == 2:
                    board[0, y, x, 1] = 1
                    if evaluate(board, 1, [y, x]):
                        board[0, y, x, 1] = 0
                        if has_winning_move:
                            has_winning_move = 2
        if not has_winning_move:
            continue

        network = Network(0)
        node = Node(0)
        expand(node, board, has_winning_move - 1, network)

        turn(board, has_winning_move - 1, node, network)

        for y in range(19):
            for x in range(19):
                if evaluate(board, has_winning_move - 1, [y, x]):
                    return True

        return False

def test_loosing_move():
    """
    test if network find winning move for the opponnet and stop it
    return true if winning move is found
    return else otherwise
    """
    while True:
        has_winning_move = 0
        tmp = np.random.randint(3, size=(19, 19))
        board = np.zeros((1, 19, 19, 3), dtype=int)

        for y in range(19):
            for x in range(19):
                if tmp[y, x] == 1:
                    board[0, y, x, 0] = 1
                    if evaluate(board, 0, [y, x]):
                        board[0, y, x, 0] = 0
                        if has_winning_move:
                            has_winning_move = 2
                elif tmp[y, x] == 2:
                    board[0, y, x, 1] = 1
                    if evaluate(board, 1, [y, x]):
                        board[0, y, x, 1] = 0
                        if has_winning_move:
                            has_winning_move = 1
        if not has_winning_move:
            continue

        network = Network(0)
        node = Node(0)
        expand(node, board, has_winning_move - 1, network)

        turn(board, has_winning_move - 1, node, network)

        for y in range(19):
            for x in range(19):
                if evaluate(board, 2 - has_winning_move, [y, x]):
                    return False

        return True

import datetime
def basic_win():
    """
    test with simple env
    """
    e = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    board = conv_map(e)
    player = 0
    pos = (6, 5)
    root = Node(0)
    net = Network(-1)
    print(evaluate(board, player, pos))
    return 1

def main():
    assert(basic_win())
    assert(test_winning_move_when_multiple())
    assert(test_winning_move_when_one())
    assert(test_loosing_move())
    pass

if __name__ ==  '__main__':
    main()

    
