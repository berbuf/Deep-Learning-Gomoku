 #!/usr/bin/env python3

"""
play a game against oneself until the end
save labels (state, policy, winning) to file label_{num_game}.label
"""

import numpy as np
from mc_tree_search import turn, expand
from node import Node

def save_tmp_label(turns, board, p, player):
    """
    append board, policy vector and current player to tmp file
    format => np.array => ((19, 19, 3), 19 * 19, 1) => 19, 19, 5
    """
    p.shape = (19,19)
    save = np.insert(board, 3, p, axis=2)
    player_layer = np.zeros((19, 19))
    player_layer[0, 0] = player
    save = np.insert(save, 4, p, axis=2)
    turns.append(save)

def save_final_label(turns, filename, winner):
    """
    rewrite tmp file and change player to 1 if == winner, else -1
    format => np.array => ((19, 19, 3), 19 * 19, 1) => 19, 19, 5
    """
    # turns.apply(lambda turn: turn[0, 0, 4] = 1 if turn[0, 0, 4] == winner else -1)
    for i in range(len(turns)):
        turns[i][0, 0, 4] = 1 if turns[i][0, 0, 4] == winner else -1
    turns = np.array(turns)
    if os.path.exists(filename):
        old_turns = np.load(filename)
        turns = np.append(old_turns, turns, axis=0)
    np.save(filename, turns)


def print_board(board):
    """
    debug, print current map to term with colours
    """
    board = (board[:,:,:,0] + board[:,:,:,1] * 2)[0]
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

def init_game(network):
    """
    init game board, first node, next player turn
    """
    node = Node(0)
    board = np.zeros((1, 19, 19, 3), np.int8)
    expand(node, board, 0, network)
    return board, node, 0

def game(network, num_game, filename):
    """
    take identifier of a game and play it until the end
    num_game: integer
    """
    turns = []
    board, root, player = init_game(network)
    end = 0
    while (not end):
        # run mcts simulation: chosen move, new board, policy, new root, game status
        _, board, p, root, end = turn(board, player, root, network)
        # save board state, policy vector and current player in tmp folder (only if it is the new player)
        if player == 1:
            save_tmp_label(turns, board, p, player)
        # next player
        player ^= 1
        # debug
        print_board(board)

    print("end game", (player ^ 1) + 1)
    # rewrite tmp file to num_game file with final winner info
    save_final_label(turns, filename, player ^ 1)