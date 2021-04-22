#!/usr/bin/env python
import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from multiagent.execution.mcts import mcts_execution
from multiagent.execution.mc_prediction import mc_prediction_execution
from multiagent.execution.q_learning import q_learning_execution
from multiagent.execution.random_policy_execution import random_policy_execution
from multiagent.execution.td3_execution import run_td3




if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    parser.add_argument('-a', '--algorithm', default='td3', help='Name of the algorithm to run. One of q_learning, td3, mcts')
    parser.add_argument('--agents', default=3, type=int, help='Number of agents')
    parser.add_argument('--landmarks', default=3, type=int, help='Number of landmarks')
    parser.add_argument('--collab', default=1, type=int, help='Toggle for shared rewards')
    
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world(agents=args.agents, landmarks=args.landmarks, collab=args.collab)
    # create multiagent environment

    # fully observable env
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)
    test_env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)

    if args.algorithm == 'td3':
        print("Running TD3")
        run_td3(env)
    if args.algorithm == 'mcts':
        mcts_execution(env)
    if args.algorithm == 'q_learning':
        q_learning_execution(env, test_env)

