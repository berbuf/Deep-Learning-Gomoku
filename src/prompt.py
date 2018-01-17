#!/usr/bin/env python3

"""
interactive prompt
"""

import argparse
import numpy as np
from node import Node
from network import Network
from reinforcement import update_turn
from mcts import mcts, expand, evaluate
from utils_board import put_on_board, print_board, print_policy

def human_turn(board, node, player, net):
    print ("Your turn, Human")

    e = 0
    while (not e):
        try:
            x = int(input("x: "))
            y = int(input("y: "))
            e = 1
        except:
            print ("wrong format, only integers required")
            q = input("quit? (y/n): ")
            if q == "y":
                exit (0)

    pos = (x, y)

    put_on_board(board, pos, player, 1)

    node = update_turn(board, player ^ 1, node, net, pos)
    _, r = evaluate(board, player, pos)

    return node, r

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-s', help='1 or 2, if you play white or black', required=True)
    parser.add_argument('-v', help='Network version', required=True)
    parser.add_argument('-d', help='1 or 0, Outputs network policy, and mcts policy', required=True)
    args = vars(parser.parse_args())

    player = int(args["s"]) - 1
    version = int(args["v"])
    debug = int(args["d"])

    board = np.zeros((19, 19, 3), np.int8)

    node = Node(0)
    net = Network(version)
    expand(node, board, player, net)


    if player == 0:
        node, _ = human_turn(board, node, player, net)

    # game
    r = 0
    while (not r):

        # ia
        if debug:
            p, _ = net.infer(board)

        _, board, p_, node, r = mcts(board, player ^ 1, node, net)
        if debug:
            print ("Network policy:")
            print_policy(board, p[0])
            print ("Mcts policy:")
            print_policy(board, p_)

        print_board(board)

        if r:
            break

        # human
        node, r = human_turn(board, node, player, net) 
        print_board(board)

        if r:
            break
       
        

        
