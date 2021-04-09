#!/usr/bin/env python
import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from multiagent.execution.mcts import mcts_execution
from multiagent.execution.q_learning import q_learning_execution
from multiagent.execution.random_policy import random_policy_execution

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)


    mcts_execution(env)
    # q_learning_execution(env)
    # random_policy_execution(env)
