#!/usr/bin/env python3

import numpy as np

class Node(object):
    """
    node object
    """
    def __init__(self, max_children):
        """
        initialize object
        """
        self._value = 0
        self._frequency = 0
        self._max_children = max_children
        self._children = dict()

    def score(self, value):
        """
        update score
        """
        self._value = value

    def add_frequency(self):
        """
        update score
        """
        self._frequency += 1

    def get_max_children(self):
        """
        getter
        """
        return self._max_children

    def get_child(self, nb_child):
        """
        return existing child node or create it
        """
        if nb_child not in self._children :
            self._children[nb_child] = Node(self._max_children - 1)
        return self._children[nb_child]

    def get_score(self):
        """
        return sum of score / frequency of all children
        if node doesn't have children, return its own score
        """
        return (self._value if not any(self._children) else
                np.sum([ node.get_score() / self._frequency for _, node in self._children.items() ]) )

    def get_policy(self):
        """
        return array of children score 
        """
        return np.array([ node.get_score() for _, node in self._children.items() ])

    def __str__(self):
        """
        print object's value
        """
        return '{}({})'.format(self.__class__.__name__, self._value)
