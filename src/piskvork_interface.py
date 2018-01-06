 #!/usr/bin/env python3

"""
implementation of piskvork protocl
piskvork is a program made for gomoku tournament
"""

import numpy as np
import threading
import os
from mcts import mcts, expand
from protocol import Protocol
from node import Node

class Thread(threading.Thread):
    def __init__(self, protocol):
        threading.Thread.__init__(self)
        self.protocol = protocol

    def run(self):
        while self.protocol.running[0]:
            self.protocol.nextCmd()

def play_turn(board, player, network):
    print("DEBUG", "Calculating the next move")
    node = Node(0)
    expand(node, board, player, network)
    (x, y), _, _, _, _ = mcts(board, player, node, network)
    print("%d,%d" % (int(x), int(y)))

def piskvork_game():
    """
    play gomoku game using trained model and piskvork interface
    """
    board = np.zeros((19, 19, 3), np.int8)
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
            play_turn(board, player, network)
        elif cmd == "begin":
            play_turn(board, player, network)
            pass
        else:
            print("ERROR")
    thread.join()
