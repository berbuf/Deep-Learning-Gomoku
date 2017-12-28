#!/usr/bin/env python3

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

    def add(self, children):
        """
        add children into node object
        """
        self._children = children

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

    def __str__(self):
        """
        print object's value
        """
        return '{}({})'.format(self.__class__.__name__, self._value)
