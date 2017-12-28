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

def evaluate():
    """
    give a score to the current state
    """

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

if __name__ == '__main__':
    board = np.zeros((3, 19, 19))
