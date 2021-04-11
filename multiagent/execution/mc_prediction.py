from collections import defaultdict
import sys
import numpy as np
from multiagent.policy.random import RandomPolicy

def mc_prediction_execution(env, num_episodes=100, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    policies = [RandomPolicy(env,i) for i in range(env.n)]
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            # probs = policy(state)
            # action = np.random.choice(np.arange(len(probs)), p=probs)
            action = []
            for i, policy in enumerate(policies):
                action.append(policy.action(state[i]))
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done[0]:
                break
            state = next_state

        # Find all states that we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key

        states_in_episode = [x[0] for x in episode]
        for state in states_in_episode:

            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if np.array(x[0]).all() == state[0].all())
            # Sum up all rewards since the first occurance
            G = sum(x[2][0] for x in episode[first_occurence_idx:])
            # Calculate average return for this state over all sampled episodes
            returns_sum[tuple(state[0])] += G
            returns_count[tuple(state[0])] += 1.0
            V[tuple(state[0])] = returns_sum[tuple(state[0])] / returns_count[tuple(state[0])]

    # print(V)
    return V