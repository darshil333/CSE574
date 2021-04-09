from multiagent.policy.policy import Policy
import numpy as np
from copy import copy
from utils.node import Node
from multiagent.algorithm.ucb import ucb
import gym
import itertools
from multiagent.multi_discrete import MultiDiscrete


playouts = 4000
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

    def action(self, env, root, best_reward):
        best_actions = []
        for _ in range(playouts):
            state = copy(env)
            (print(s) for s in state.action_space)
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
                _, reward, terminal, _ = state.step(node.action)
                sum_reward += reward
                actions.append(node.action)

            # expansion
            if not terminal:
                node.children = [Node(node, a)
                                 for a in combinations(state.action_space)]
                np.random.shuffle(node.children)

            # playout
            while not terminal:
                action = MultiDiscrete.sample(self)
                _, reward, terminal, _ = state.step(action)
                sum_reward += reward
                actions.append(action)

                if len(actions) > max_depth:
                    sum_reward -= 100
                    break

            # remember best
            if best_reward < sum_reward:
                best_reward = sum_reward
                best_actions = actions

            # backpropagate
            while node:
                node.visits += 1
                node.value += sum_reward
                node = node.parent

        return best_actions
