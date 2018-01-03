import mc_tree_search

class Protocol:
    def __init__(self, b, r):
        self.infos = {"timeout_turn": 4000, "timeout_match": 0, "max_memory": 70000,
         "time_left": 2147483647, "game_type": 1, "rule": 1,
         "evaluate": [0, 0], "folder": "/tmp/"}
        self.board = b
        self.running = r
        self.cmdTab = []

    def nextCmd(self):
        line = input()
        line = line[:-1] if line[-1:] == "\n" else line
        args = line.split(' ')
        cmd = args[0].lower()
        del args[0]
        if cmd == "board":
            new_board = []
            inp = input()
            inp = inp[:-1] if inp[-1:] == "\n" else inp
            while inp.lower() != "done":
                new_board.append(inp)
                inp = input()
                inp = inp[:-1] if inp[-1:] == "\n" else inp
            moves = []
            for line in new_board:
                move = line.split(',')
                pos = [0, 0, 0]
                if (len(move) != 3):
                    print("ERROR")
                    break
                try:
                    pos[0] = int(move[0])
                    pos[1] = int(move[1])
                    pos[2] = int(move[2])
                except:
                    print("ERROR")
                    break
                moves.append(pos)
            lastMove = 0
            for move in moves:
                x = move[0];
                y = move[1];
                player = move[2] % 2; # map 2 -> 0 and 1 -> 1
                put_on_board(self.board, (x, y), player, 0)
                # self.board[0][move[1]][move[0]] = move[2]
                print("DEBUG", "Player", move[2], ":", move[0], move[1])
                lastMove = move[2]
            if lastMove == 2:
                self.cmdTab.append("begin")
        elif cmd == "info":
            try:
                if args[0].lower() in self.infos:
                    self.infos[args[0].lower()] = args[1]
                    print("DEBUG", args[0].lower(), "set to", args[1])
            except:
                print("ERROR")
        elif cmd == "end":
            self.running[0] = 0
        elif cmd == "about":
            print('name="Alpha", version="2.0", author="DreamTeam", country="France"')
        else:
            self.cmdTab.append(line)

    def pullCmd(self):
        try:
            cmd = self.cmdTab[0].split(' ')
            del self.cmdTab[0]
            return cmd
        except:
            return ["none"]
