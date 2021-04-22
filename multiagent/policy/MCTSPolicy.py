from multiagent.policy.policy import Policy
import numpy as np
from copy import copy
from utils.node import Node
from multiagent.algorithm.ucb import ucb
import gym
import itertools


playouts = 10
max_depth = 50


def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    else:
        return itertools.product(*[combinations(s) for s in space])

class MCTSPolicy(Policy):
    def __init__(self, env, agent_index):
        super(MCTSPolicy, self).__init__()
        self.env = env

    def action(self, env, agent_index):
        best_actions = []
        best_reward = float("-inf")
        root = Node(None, None)
        for _ in range(playouts):
            state = copy(env)
            sum_reward = 0
            node = root
            terminal = False
            actions = []

            # selection
            while node.children:
                if node.explored_children < len(node.children):
                    child = node.children[node.explored_children]
                    node.explored_children += 1
                    node = child
                else:
                    node = max(node.children, key=ucb)
                _, reward, terminal, _ = state.single_step([node.action], agent_index)
                sum_reward += reward
                actions.append(node.action)

            # expansion
            if not terminal:
                node.children = [Node(node, a)
                                 for a in combinations(state.action_space[agent_index])]
                np.random.shuffle(node.children)

            # playout
            while not terminal:
                action = state.action_space[agent_index].sample()
                _, reward, terminal, _ = state.single_step([action], agent_index)
                sum_reward += reward
                actions.append(action)
                if len(actions) > max_depth:
                    sum_reward -= 100
                    break
            # remember best
            # print(best_reward, sum_reward)
            if best_reward < sum_reward:
                best_reward = sum_reward
                best_actions = actions

            # backpropagate
            while node:
                node.visits += 1
                node.value += sum_reward
                node = node.parent
        return best_actions
