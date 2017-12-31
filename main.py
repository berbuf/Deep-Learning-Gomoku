#!/usr/bin/env python3

import numpy as np
from game import game
import protocol
import threading

class Thread(threading.Thread):
    def __init__(self, protocol):
        threading.Thread.__init__(self)
        self.protocol = protocol

    def run(self):
        while protocol.running[0]:
            self.protocol.nextCmd()

if __name__ == '__main__':
    # parameters
    number_of_games = 1

    for num_game in range(number_of_games):

        # play a game until the end
        game(num_game)

        # train network
        #train()
    """
    # Piskvork
    board = np.zeros((3, 19, 19), np.int8)
    running = [1]
    protocol = protocol.Protocol(board, running)
    thread = Thread(protocol)

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
                board[0][int(pos[1])][int(pos[0])] = '2'
            except:
                print("ERROR")
            print("DEBUG", "Calculating the next move")
            # ADD OWN TURN CALCULATION
            # print(x + "," + y)
        elif cmd == "begin":
            print("DEBUG", "Calculating the next move")
            # ADD OWN TURN CALCULATION
            # print(x + "," + y)
            pass
        else:
            print("ERROR")
    thread.join()
    """
