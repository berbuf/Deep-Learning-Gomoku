 #!/usr/bin/env python3

import numpy as np
from game import game

if __name__ == '__main__':

    # parameters
    number_of_games = 1

    for num_game in range(number_of_games):

        # play a game until the end
        game(num_game)

        # train network
        #train()
