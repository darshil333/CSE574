import sys
from time import time

from multiagent.policy.MCTSPolicy import MCTSPolicy
from utils.node import Node
import numpy as np

loops = 1000


def moving_average(v, n):
    n = min(len(v), n)
    ret = [.0]*(len(v)-n+1)
    ret[0] = float(sum(v[:n]))/n
    for i in range(len(v)-n):
        ret[i+1] = ret[i] + float(v[n+i] - v[i])/n
    return ret


def print_stats(loop, score_1, score_2, avg_time):
    sys.stdout.write('\r%3d   agent_1_moving_score:%10.3f  agent_2_moving_score:%10.3f  avg_time:%4.1f s' %
                     (loop, score_1, score_2, avg_time))
    sys.stdout.flush()


def mcts_execution(env):
    best_rewards_1 = []
    best_rewards_2 = []
    start_time = time()

    policies = [MCTSPolicy(env, i) for i in range(env.n)]
    
    total_reward = np.zeros(env.n)
    
    for loop in range(loops):
        env.reset()
        best_actions = []
        for i, policy in enumerate(policies):
            best_actions.append(policy.action
                                (env, i))
        sum_reward_1 = 0
        sum_reward_2 = 0
        # next_state, reward, done, info = env.step(best_actions)
        # sum_reward_1 += reward[0]
        # sum_reward_2 += reward[1]
        for action in best_actions:
            _, reward, terminal, _ = env.step(action)
            sum_reward_1 += reward[0]
            sum_reward_2 += reward[1]
            if terminal:
                break
        env.render()

        # display rewards
        # for i, agent in enumerate(env.world.agents):
        #     print(agent.name + " reward: %0.3f" % env._get_reward(agent, i))

        for agent in range(env.n):
            total_reward[agent] += reward[agent]

        # best_rewards_1.append(sum_reward_1)
        # best_rewards_2.append(sum_reward_2)
        # score_1 = max(moving_average(best_rewards_1, 100))
        # score_2 = max(moving_average(best_rewards_2, 100))
        # avg_time = (time()-start_time)/(loop+1)
        # print_stats(loop+1, score_1, score_2, avg_time)

    print("Total value of episode for each agent - ", total_reward)