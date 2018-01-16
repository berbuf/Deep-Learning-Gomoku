#!/usr/bin/env python3

"""
perform a monte carlo tree search for a given board
with the help of a policy and value network
"""

import numpy as np
from node import Node
from utils_board import update_board_player, put_on_board, get_pos_on_board

THREATS = [
    ('\x01\x01\x01\x01\x01', 1),
    ('\x00\x01\x01\x01\x01\x00', 0.2),
    ('\x01\x01\x01\x01\x00', 0.2),
    ('\x00\x01\x01\x01\x01', 0.2),
    ('\x01\x01\x01\x00\x01', 0.2),
    ('\x01\x00\x01\x01\x01', 0.2),
    ('\x01\x01\x00\x01\x01', 0.2),
    ('\x01\x01\x01\x00\x00', 0.1),
    ('\x00\x00\x01\x01\x01', 0.1),
    ('\x01\x01\x00\x00\x01', 0.1),
    ('\x01\x00\x00\x01\x01', 0.1),
    ('\x01\x01\x00\x01\x00', 0.1),
    ('\x00\x01\x00\x01\x01', 0.1),
    ('\x01\x00\x01\x00\x01', 0.1),
    ('\x00\x01\x01\x01\x00', 0.1),
]

def evaluate(board, player, pos):
    """
    give a score to the current state
    board => (19, 19, 3)
    player => 0 or 1
    pos => (x, y)
    """

    score = 0

    maps = [ board[:,:,0], board[:,:,1] ]
    pmap = maps[player]

    for threat, value in THREATS:
        line = pmap[pos[1]] # get line at pos
        p = pos[0]
        score += value * (threat in ''.join(map(chr, line[max(p-5, 0):p+5])))

        line = pmap[:,pos[0]] # get column at pos
        p = pos[1]
        score += value * (threat in ''.join(map(chr, line[max(p-5, 0):p+5])))

        line = pmap.diagonal(pos[0] - pos[1]) # get diagonal 1
        p = min(pos[0], pos[1])
        score += value * (threat in ''.join(map(chr, line[max(p-5, 0):p+5])))

        line = np.fliplr(pmap).diagonal(18 - pos[0] - pos[1]) # get diagonal 2
        p = min(pos[1], 18 - pos[0])
        score += value * (threat in ''.join(map(chr, line[max(p-5, 0):p+5])))

    return score

def expand(node, board, player, network):
    """
    expand node
    predict state value, and probability for children using neural network
    """
    update_board_player(board, player)
    # dummy values
    """
    p, v = (np.array( [ 1.5 for i in range(19 * 19) ] ), 0)
    p = p[np.where((board[:,:,0] + board[:,:,1]).flatten() == 0)]
    # run network (time consuming)
    """
    p, v = network.infer(board)
    # remove already played tiles
    p = p[0][np.where((board[:,:,0] + board[:,:,1]).flatten() == 0)]
    # negative if black
    p *= (1 - 2 * player)
    v *= (1 - 2 * player)
    node.expand_children(p)
    return v

def select(node, board, player):
    """
    return chosen node, updated board, new coordinates
    """
    # choose next node
    n = np.argmax(node.get_policy() * (1 - 2 * player))
    child = node.get_child(n)
    child.add_frequency()
    # get coordinates of next move, and update board
    x, y = get_pos_on_board(board, n)
    put_on_board(board, (x, y), player, 1)
    return child, board, (x, y), player ^ 1

def search(node, board, player, network):
    """
    node: object Node
    board: np.array(3,19,19)
    player: 0 for white, 1 for black
    do actions on a level of deepness
    """
    child, board, pos, next_player = select(node, board, player)
    # evaluate or keep searching
    if child.leaf():
        value = evaluate(board, player, pos)
        # if not a winning move
        if not value:
            value = expand(child, board, next_player, network)
    else:
        value = search(child, board, next_player, network)
    # clean board and back propagate
    put_on_board(board, pos, player, 0)
    child.score(value)
    return value

def mcts(board, player, root, network):
    """
    board: np.array((3, 19, 19))
    take board, player turn (0, 1), root node
    return next move, updated board, policy vector, next root and boolean for game status
    """

    # parameters: number of search
    trials = 1000

    # build tree
    for _ in range(trials):
        search(root, board, player, network)

    # reshape policy to (361)
    p = np.ones(361) - (board[:,:,0] + board[:,:,1]).flatten()
    p[ p == 1 ] = root.get_mcts() * (1 - 2 * player)
    print (p)

    # get coordinates of chosen move, and update board
    n = root.get_max_frequency_move()
    x, y = get_pos_on_board(board, n)
    put_on_board(board, (x, y), player, 1)

    return ((x, y), board, p, root.get_child(n), evaluate(board, player, (x, y)))
