from multiagent.policy.td3 import run
import matplotlib.pyplot as plt
import os
import numpy as np

os.makedirs("./results/plots/", exist_ok=True)
os.makedirs ("./results/plots/td3/", exist_ok=True)

def run_td3(env):
    model_path = './results/td3/'
    hidden_dim = 256
    rendermode = False
    episodes = 200
    steps_per_episode = 100
    warmup = 5000
    batchsize = 512
    train_rewards = run(env,
                        model_path,
                        hidden_dim,
                        batch=batchsize,
                        train=True,
                        warmup=warmup,
                        render=rendermode,
                        episodes=episodes,
                        steps_per_episode=steps_per_episode)
    test_rewards = run(env,
                       model_path,
                       hidden_dim,
                       train=False,
                       batch=batchsize,
                       warmup=warmup,
                       render=rendermode,
                       episodes=episodes,
                       steps_per_episode=steps_per_episode)

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(range(len(train_rewards)), train_rewards, label="Average Reward")  # Plot some data on the axes.
    ax.set_xlabel('Episodes')  # Add an x-label to the axes.
    ax.set_ylabel('Reward')  # Add a y-label to the axes.
    ax.set_title("Cooperative Agents")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.savefig("./results/plots/td3/train_rewards_collab.png")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(range(len(test_rewards)), test_rewards, label='Average Reward')  # Plot some data on the axes.
    ax.set_xlabel('Episodes')  # Add an x-label to the axes.
    ax.set_ylabel('Average Episodic Reward')  # Add a y-label to the axes.
    ax.set_title("Cooperative Agents")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.savefig("./results/plots/td3/test_rewards_collab.png")
