import numpy as np
import random

class TictactoeEnv:
    def __init__(self):
        self.win_reward = 1
        self.invalid_move_reward = -1
        self.default_reward = 0
        self.reset()

    def is_valid_move(self, y, x):
        if self.board[y][x] == 0:
            return True
        else:
            return False

    def gen_board(self):
        return np.zeros((3,3))

    def reset(self):
        self.board = self.gen_board()
        self.done = False
        self.turn = 'x'

        return self.encode_board()

    def encode_board(self):
        encoded_board = np.zeros((2,3,3))

        #   o's
        for y in range(3):
            for x in range(3):
                if not self.board[y][x] == 1:
                    encoded_board[0][y][x] = 1

        #   x's
        for y in range(3):
            for x in range(3):
                if not self.board[y][x] == 2:
                    encoded_board[1][y][x] = 1

        return encoded_board

    def moves_left(self):
        for y in range(3):
            for x in range(3):
                if self.board[y][x] == 0:
                    return True
        return False

    def toggle_player(self):
        if self.turn == 'x':
            self.turn = 'o'
        else:
            self.turn = 'x'

    def get_winner(self):
        diagPositions = [
            [   [0, 0],
                [1, 1],
                [2, 2]],
            [   [0, 2],
                [1, 1],
                [2, 0]]]

        for player in [1, 2]:
            #   check horizontal
            for row in self.board:
                count = 0
                for col in row:
                    if col == player:
                        count += 1
                if count == 3:
                    return player
            #   check vertical
            for c in range(0, 3):
                count = 0
                for r in range(0, 3):
                    piece = self.board[r][c]
                    if piece == player:
                            count += 1
                if count == 3:
                    return player

            for direction in diagPositions:
                count = 0
                for pos in direction:
                    x = pos[0]
                    y = pos[1]
                    piece = self.board[y][x]
                    if piece == player:
                            count += 1
                if count == 3:
                    return player
        return False

    def step(self, action, player):
        info = {"player": player,
                "invalid_move": False}

        game_player = 2 if player == 0 else 1

        done = False
        state_ = self.board
        reward = self.default_reward

        if self.done:
            state_ = self.encode_board()
            return state_, 0, self.done, info
        print("donecheck {}".format(done))

        move = action.argmax()
        y = move // 3
        x = move % 3

        if not self.is_valid_move(y, x):
            info["invalid_move"] = True
            print("invalid move")
            state_ = self.encode_board()
            return state_, self.invalid_move_reward, self.done, info

        self.board[y][x] = 1 if self.turn == 'o' else '2'
        self.toggle_player()

        winner = self.get_winner()
        if winner:
            self.done = True
            reward = self.win_reward
        print("player win {}".format(self.done))
        
        if not self.moves_left():
            self.done = True
        print("moves left {}".format(self.done))

        state_ = self.encode_board()

        return state_, reward, self.done, info

    def render(self):
        self.urmom()

    def urmom(self):
        self.print_board()

    def print_board(self):
        print("= = =")
        for y in range(3):
            if not y == 0:
                print()
            for x in range(3):
                urmom = self.board[y][x]
                piece = '-'
                if urmom == 1:
                    piece = 'O'
                elif urmom == 2:
                    piece = 'X'
                print(piece + " ",end='')
        print()
        print("= = =")
        
if __name__ == '__main__':
    env = TictactoeEnv()
    state = env.reset()

    player = 0

    steps = 0

    done = False
    while not done:
        print()
        print("steps {}".format(steps))
        env.render()

        random_move = random.randint(0, 8)
        action = np.zeros(9)
        action[random_move] = 1.0
        print("action {}".format(random_move))

        state_, reward, done, info = env.step(action, player)
        print("reward: {}, done: {}, info: {}".format(reward, done, info))

        state = state_
        if not info["invalid_move"]:
            if player == 1:
                player = 0
            else:
                player = 1
        
        steps += 1
    print()
    print()
    print("GAME OVER")
    env.render()