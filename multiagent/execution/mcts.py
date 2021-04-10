import sys
from time import time

from multiagent.policy.MCTSPolicy import MCTSPolicy
from utils.node import Node

loops = 100


def moving_average(v, n):
    n = min(len(v), n)
    ret = [.0]*(len(v)-n+1)
    ret[0] = float(sum(v[:n]))/n
    for i in range(len(v)-n):
        ret[i+1] = ret[i] + float(v[n+i] - v[i])/n
    return ret


def print_stats(loop, score, avg_time):
    sys.stdout.write('\r%3d   score:%10.3f   avg_time:%4.1f s' %
                        (loop, score, avg_time))
    sys.stdout.flush()


def mcts_execution(env):
    best_rewards = []
    start_time = time()

    policies = [MCTSPolicy(env, i) for i in range(env.n)]

    for loop in range(loops):
        env.reset()
        root = Node(None, None)

        sample_actions = env.sample_actions()
        best_actions = []
        for i, policy in enumerate(policies):
            best_actions = (policy.action
                (env, root, [sample_actions[i]]))

        # next_state, reward, done, info=env.step(best_actions)

        sum_reward=0
        for action in best_actions:
            _, reward, terminal, _ = env.step(action)
            sum_reward += reward[0]
            if terminal:
                break

        env.render()

        # # display rewards
        # for agent in env.world.agents:
        #     print(agent.name + " reward: %0.3f" % env._get_reward(agent))
        best_rewards.append(sum_reward)
        print(best_rewards)  
        score=max(moving_average(best_rewards, 100))
        avg_time=(time()-start_time)/(loop+1)
        print_stats(loop+1, score, avg_time)
