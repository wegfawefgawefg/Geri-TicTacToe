import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from agent import Agent

import numpy as np
import os
import datetime

import tictactoe_env as ttt

'''
investigate how many times each player wins
    the bot should tie every game

print out the q values and turns for each render

assert that 0 is learning anything at all

graph the loss to see if its going down

enable checkpoints and create a seperate play file to vs the bot



'''


if __name__ == '__main__':
    TENSOR_BOARD = False
    SAVE_CHECKPOINTS = False
    CHECKPOINT_INTERVAL = 1000
    LR = 0.001

    if TENSOR_BOARD:
        ENV_NAME = "Tictactoe"
        DISC_OR_CONT = "Disc"
        ALGO_NAME = "DQN"

        YMDHMS = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        CHECKPOINT_NAME = "_".join([ENV_NAME, DISC_OR_CONT, ALGO_NAME, str(LR)])
        RUN_NAME = "_".join([YMDHMS, CHECKPOINT_NAME])
        RUNS_PATH = os.path.join("..", "runs", "tictactoe", RUN_NAME)
        writer = SummaryWriter(RUNS_PATH, comment=CHECKPOINT_NAME)

    #   MAKE ENV
    env = ttt.TictactoeEnv()
    agent = Agent(lr=LR, state_shape=(3,3,3), num_actions=9, batch_size=64)

    NUM_EPISODES = 10000

    #   STATS
    numHiddenEpisodes = 500
    high_score = -math.inf

    episode = 0
    num_samples = 0
    while True:
        done = False
        state = env.reset()
        player = 0

        if episode > NUM_EPISODES:
            break

        score, frame = 0, 1
        while not done:
            if episode > numHiddenEpisodes:
                env.render()

            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action, player)
            invalid_move = True if info["invalid_move"] else False
            player = info["player"]
            # print(player)

            agent.store_memory(state, action, reward, state_, done, 
                invalid_move)
            agent.learn()

            state = state_

            score += reward
            frame += 1
            num_samples += 1

        high_score = max(high_score, score)

        print(( "num-samples: {}, ep {}: high-score {:12.3f}, "
                "score {:12.3f}, epsilon {:6.3f}").format(
            num_samples, episode, high_score, score, agent.epsilon.value()))

        if SAVE_CHECKPOINTS:
            if episode % CHECKPOINT_INTERVAL == 0:
                print("SAVING CHECKPOMT")
                checkpointName = "%s_%d_%d.dat" % (CHECKPOINT_NAME, episode, score)
                fname = os.path.join('.', 'checkpoints', checkpointName)
                torch.save( agent.dqn.state_dict(), fname )

        episode += 1