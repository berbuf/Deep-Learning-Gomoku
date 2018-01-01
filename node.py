#!/usr/bin/env python3

"""
class node, implement a level of deepness in a certain path for the monte carlo tree search
"""

import numpy as np

class Node(object):
    """
    node object
    """
    def __init__(self, probability):
        """
        initialize object
        """
        self._value = 0
        self._frequency = 0
        self._probability = probability
        self._children = dict()

    def get_value(self):
        """
        getter
        """
        return self._value

    def get_frequency(self):
        """
        getter
        """
        return self._frequency

    def score(self, value):
        """
        update value
        """
        self._value = value

    def add_frequency(self):
        """
        update score
        """
        self._frequency += 1

    def leaf(self):
        """
        check if node has been explored
        """
        return not any(self._children)

    def get_score(self):
        """
        getter
        """
        return 0 if not self._frequency else self._value / self._frequency

    def get_probability(self):
        """
        return probability / 1 + frequency
        """
        return self._probability / (1 + self._frequency)

    def get_policy(self):
        """
        return array of children value for Q + U
        Q => value / frequency
        U => probability / 1 + frequency
        """
        return np.array([ node.get_score() + node.get_probability()
                          for _, node in self._children.items() ])

    def expand_children(self, p):
        """
        initialize children dict with array of probabilities
        """
        self._children = { n: Node(v) for n, v in enumerate(p) }

    def get_child(self, nb_child):
        """
        return child node
        """
        return self._children[nb_child]

    def get_max_frequency_move(self):
        """
        return max visited node
        """
        return np.argmax( [ node.get_frequency() for _, node in self._children.items() ] )

    def debug(self):
        print (len(self._children))
        print ( np.sum( [ node.get_frequency() for _, node in self._children.items() ] ) )
        print ( [ node.get_frequency() for _, node in self._children.items() ] )

    def __str__(self):
        """
        print object's value
        """
        return '{}({})'.format(self.__class__.__name__, self._value)
