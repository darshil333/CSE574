from collections import defaultdict
import numpy as np
# from multiagent.environment import MultiAgentEnv
from multiagent.policy.epsilon_greedy import EpsilonGreedyPolicy
from multiagent.algorithm.q_learning_update import q_learning_update


def q_learning_execution(env, test_env):
    # q learning initializations

    # dictionary that maps from state, s, to a numpy array of Q values [Q(s, a_1), Q(s, a_2) ... Q(s, a_n)] and everything is initialized to 0.
    q_vals = defaultdict(lambda: np.array([0. for _ in range(4)]))  # env.action_space.n = 4
    gamma = 0.95
    alpha = 0.1

    epsilon = 1
    max_epsilon = 1
    min_epsilon = 0.01
    explortion_decay = 0.01

    def greedy_eval():
        # evaluate greedy policy w.r.t current q_vals
        prev_state = test_env.reset()
        ret = 0.
        done = False
        H = 100
        for i in range(H):
            action = []
            for agent in range(env.n):
                action.append(np.argmax(q_vals[tuple(prev_state[agent])]))
            state, reward, done, info = test_env.step(action)
            ret += reward[0]
            prev_state = state
        return ret / H

    # create policies for each agent
    # env.n = number of agents
    policies = [EpsilonGreedyPolicy(env, i) for i in range(env.n)]

    num_episodes = 10000
    max_steps_per_episode = 100
    
    all_episode_reward = np.zeros((num_episodes,env.n))

    for episode in range(num_episodes):
        current_state = env.reset()
        done = False
        current_episode_reward = np.zeros(env.n)
        for step in range(max_steps_per_episode):
            # select action using eps_greedy for each agent
            action = []
            for agent, policy in enumerate(policies):
                action.append(policy.action(tuple(current_state[agent]), q_vals, epsilon))
            
            # run action
            # print('Actions for each agent in step {} -- {}'.format(step, action))
            next_state, reward, done, info = env.step(action)
            # print("Reward in episode {} step {} -- {}".format(episode, step, reward))

            # update q value
            for agent in range(env.n):
                q_learning_update(gamma, alpha, q_vals, tuple(current_state[agent]), action, tuple(next_state[agent]), reward)
                # print("Reward of {} -- {}".format(agent, reward[agent]))
                current_episode_reward[agent] += reward[agent]
            
            current_state = next_state

            if done == True:
                break
        
        # Decay exploration rate
        epsilon = min_epsilon + \
            (max_epsilon - min_epsilon) * np.exp(-explortion_decay * episode)

        for agent in range(env.n):
            all_episode_reward[episode, agent] = current_episode_reward[agent]

        # render all agent views
        env.render()

        # # evaluation
        # if episode % 1000 == 0:
        #     print("Episode %i # Average reward: %.2f" % (episode, greedy_eval()))

    # Average reward per thousand episodes
    reward_per_thousand_episodes = np.split(np.array(all_episode_reward), num_episodes/1000)
    count = 1000
    print("----Average reward per thousand episodes---\n")
    for r in reward_per_thousand_episodes:
        print(count, ': ', str(sum(r/1000)))
        count += 1000