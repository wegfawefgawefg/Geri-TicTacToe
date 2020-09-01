import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import datetime

import tictactoe_env as ttt

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
    agent = Agent(lr=LR, stateShape=(18,), numActions=9, batchSize=64)

    NUM_SAMPLES = 100

    #   STATS
    numHiddenEpisodes = -1
    high_score = -math.inf

    episode = 0
    num_samples = 0
    while True:
        done = False
        state = env.reset()

        if num_samples > NUM_SAMPLES:
            break

        score, frame = 0, 1
        while not done:
            # if episode > numHiddenEpisodes:
            #     env.render()

            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            agent.store_memory(state, action, reward, state_, done)
            agent.learn()

            state = state_

            score += reward
            frame += 1
            num_samples += 1

        high_score = max(high_score, score)

        print(( "num-samples: {}, ep {}: high-score {:12.3f}, "
                "score {:12.3f}, epsilon {:6.3f}").format(
            num_samples, episode, high_score, score, agent.epsilon))

        if SAVE_CHECKPOINTS:
            if episode % CHECKPOINT_INTERVAL == 0:
                print("SAVING CHECKPOMT")
                checkpointName = "%s_%d_%d.dat" % (CHECKPOINT_NAME, episode, score)
                fname = os.path.join('.', 'checkpoints', checkpointName)
                torch.save( agent.dqn.state_dict(), fname )

        episode += 1