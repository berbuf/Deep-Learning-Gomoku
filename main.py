#!/usr/bin/env python3

import numpy as np


class Node(object):
    """
    node object
    """
    def __init__(self, value):
        """
        initialize object
        """
        self.value = value
        self._childs = None

    def add(self, childs):
        """
        add childs into node object
        """
        self._childs = childs

    def __str__(self):
        """
        print object's value
        """
        return '{}({})'.format(self.__class__.__name__, self.value)


def mc_search():
    """
    do actions on a level of deepness
    """

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


def policy():
    """
    establish scores to all possible action
    """

def turn():
    """
    take a state as input, and return a position 
    """
    p = mc_search(3, board)
    policy(p)

def test_evaluate():
    while input('press q to quit: ') != 'q':
        tmp = np.random.randint(3, size=(19, 19))
        board = np.zeros((3, 19, 19), dtype=int)
        for y in range(19):
            for x in range(19):
                if tmp[y, x] == 1:
                    board[0, y, x] = 1
                elif tmp[y, x] == 2:
                    board[1, y, x] = 1

        wpos = []
        for y in range(19):
            for x in range(19):
                if evaluate(board, [y, x]):
                    wpos.append([y, x])
        board[2] = np.full((19, 19), 1, dtype=int)
        for y in range(19):
            for x in range(19):
                if evaluate(board, [y, x]):
                    wpos.append([y, x])

        for y in range(19):
            for x in range(18):
                v = board[0, y, x] + board[1, y, x] * 2
                if v:
                    if [y, x] in wpos:
                        print('\033[92m{}\033[0m'.format(v), end='')
                    else:
                        print(v, end='')
                else:
                    print('0', end='')
            print()
    pass

if __name__ == '__main__':
    board = np.zeros((3, 19, 19))
    test_evaluate()
