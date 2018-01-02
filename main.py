#!/usr/bin/env python3

import numpy as np
from reinforcement import game
from network import Network
import mc_tree_search
from protocol import Protocol
from node import Node
import threading

class Thread(threading.Thread):
    def __init__(self, protocol):
        threading.Thread.__init__(self)
        self.protocol = protocol

    def run(self):
        while self.protocol.running[0]:
            self.protocol.nextCmd()

def play_turn(board, player, network):
    # ADD OWN TURN CALCULATION
    node = mc_tree_search.Node(0)
    mc_tree_search.expand(node, board, 0, network)
    (x, y), _, _, _, _ = mc_tree_search.turn(board, player, node, network)
    print(x + "," + y)

def piskvork_game():
    """
    play gomoku game using trained model and piskvork interface
    """
    board = np.zeros((1, 19, 19, 3), np.int8)
    running = [1]
    protocol = Protocol(board, running)
    thread = Thread(protocol)
    network = Network(-1)
    player = 0

    thread.start()
    while running[0]:
        args = protocol.pullCmd();
        cmd = args[0].lower()
        del args[0]
        if cmd == "none":
            pass
        elif cmd == "start":
            print("OK")
        elif cmd == "turn":
            try:
                pos = args[0].split(',')
                x = int(pos[0])
                y = int(pos[1])
                mc_tree_search.put_on_board(board, (x, y), player ^ 1, 0)
            except:
                print("ERROR")
            print("DEBUG", "Calculating the next move")
            play_turn(board, player, network)
        elif cmd == "begin":
            print("DEBUG", "Calculating the next move")
            play_turn(board, player, network)
            pass
        else:
            print("ERROR")
    thread.join()

def reinforcement():
    """
    train model against itself
    """

    # parameters
    number_of_games = 1
    network = Network(-1)

    for num_game in range(number_of_games):
        # play a game until the end
        game(network, num_game)

        # train network
        network.train()

if __name__ == '__main__':
    # # training
    # reinforcement()

    # real game
    piskvork_game()
