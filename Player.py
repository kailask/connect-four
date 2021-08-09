from multiprocessing import Value
import numpy as np


class AIPlayer:
    def __init__(self, player_number, debug):
        self.player_number = player_number
        self.other_player_number = 1 if player_number == 2 else 2
        self.type = 'ai'
        self.player_string = 'Player {} AI'.format(player_number)
        self.MAX_DEPTH = 3
        self.debug = debug

    def get_alpha_beta_move(self, board):
        best_move = (-1, -np.inf)
        for i, s in self.enumerate_states(board, self.player_number):
            move = self.get_alpha_beta_value(s, False, 0, best_move[1], np.inf)
            self.debug and print((i, move))
            best_move = max((i, move), best_move, key=lambda m: m[1])
        print('{}: {}'.format(self.player_string, best_move[0]))
        return best_move[0]

    def get_alpha_beta_value(self, board, my_turn, depth, a, b):
        current_player_number = self.current_player_number(my_turn)
        if depth == self.MAX_DEPTH or self.is_terminal(board, my_turn):
            return self.evaluation_function(board, my_turn)

        if my_turn:
            max_move = -np.inf
            for _, s in self.enumerate_states(board, current_player_number):
                move = self.get_alpha_beta_value(s, not my_turn, depth+1, a, b)
                if move > max_move:
                    max_move = move
                    a = max(max_move, a)
                if move >= b:
                    return np.inf
            return max_move
        else:
            min_move = np.inf
            for _, s in self.enumerate_states(board, current_player_number):
                move = self.get_alpha_beta_value(s, not my_turn, depth+1, a, b)
                if move < min_move:
                    min_move = move
                    b = min(min_move, b)
                if move <= a:
                    return -np.inf
            return min_move

    def get_expectimax_move(self, board):
        moves = [(i, self.get_expectimax_value(s, False, 0))
                 for i, s in self.enumerate_states(board, self.player_number)]
        self.debug and print(moves)
        move = max(moves, key=lambda m: m[1])[0]

        print('{}: {}'.format(self.player_string, move))
        return move

    def get_expectimax_value(self, board, my_turn, depth):
        if depth == self.MAX_DEPTH or self.is_terminal(board, my_turn):
            return self.evaluation_function(board, my_turn)

        states = self.enumerate_states(
            board, self.current_player_number(my_turn))
        moves = [self.get_expectimax_value(
            s, not my_turn, depth+1) for _, s in states]
        return max(moves) if my_turn else sum(moves)/len(moves)

    def enumerate_states(self, board, player_number):
        states = []
        for c in range(len(board[0])):
            for r in range(len(board)-1, -1, -1):
                if board[r][c] == 0:
                    s = board.copy()
                    s[r][c] = player_number
                    states.append((c, s))
                    break
        return states

    def current_player_number(self, my_turn):
        return self.player_number if my_turn else self.other_player_number

    def prev_player_number(self, my_turn):
        return self.player_number if not my_turn else self.other_player_number

    def evaluation_function(self, board, my_turn):
        eval_sum = 0

        # Check for win
        if self.player_won(board, self.prev_player_number(my_turn)):
            eval_sum = 100000 if not my_turn else -100000
        elif np.all(board[0]):
            return 0

        # Check for points
        points = self.count_points(board)
        eval_sum += points[self.player_number]
        eval_sum -= points[self.other_player_number]

        return eval_sum

    def count_points(self, board):
        def calc_points(windows):
            points = [0, 0, 0]
            for count in [np.pad(np.bincount(w), (0, 2)) for w in windows]:
                if count[1] == 0:
                    points[2] += count[2]**6
                if count[2] == 0:
                    points[1] += count[1]**6
            return points

        def check_horizontal(b):
            windows = [row[i:i+4] for row in b for i in range(len(row)-3)]
            return calc_points(windows)

        def check_vertical(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            diagonals = []
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                diagonals.append(np.diagonal(op_board, offset=0))
                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diagonals.append(np.diagonal(op_board, offset=offset))
            windows = [d[i:i+4] for d in diagonals for i in range(len(d)-3)]
            return calc_points(windows)

        return np.add(np.add(check_horizontal(board), check_vertical(board)), check_diagonal(board))

    def is_terminal(self, board, my_turn):
        return np.all(board[0]) or self.player_won(board, self.prev_player_number(my_turn))

    def player_won(self, board, player_num):
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        def to_str(a): return ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {} Random'.format(player_number)

    def get_move(self, board):
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        move = np.random.choice(valid_cols)
        print('{}: {}'.format(self.player_string, move))
        return move


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {} Human'.format(player_number)

    def get_move(self, board):
        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        while True:
            try:
                move = int(input(self.player_string + " move: "))
                if move not in valid_cols:
                    raise Exception('Invalid column')
                return move
            except:
                print('Choose a column from: {}'.format(valid_cols))
