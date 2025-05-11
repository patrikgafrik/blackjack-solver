import gymnasium as gym
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

# Define optimal strategy matrices
optimal_hard_totals = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

optimal_soft_totals = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


def get_decision(strategy, player_sum, dealer_upcard, is_soft):
        hard_totals, soft_totals = strategy
        dealer_idx = dealer_upcard - 2  # Adjust dealer card index to match array indexing
    
        if is_soft == 1:
            row_index = max(0, min(20 - player_sum, soft_totals.shape[0] - 1))
            return soft_totals[row_index, dealer_idx]
        else:
            row_index = max(0, min(20 - player_sum, hard_totals.shape[0] - 1))
            return hard_totals[row_index, dealer_idx]  # Adjusted index for hard_totals
        
# Runs N episodes in a single process
def play_episodes_batch(strategy, episodes):
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            player_sum, dealer_card, usable_ace = obs
            if dealer_card == 1:
                dealer_card = 11
            action = get_decision(strategy, player_sum, dealer_card, usable_ace)
            obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    env.close()
    return total_reward

# Divide total episodes among processes
def evaluate_strategy(strategy, total_episodes=10000, workers=8):
    episodes_per_worker = total_episodes // workers
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rewards = executor.map(play_episodes_batch, [strategy] * workers, [episodes_per_worker] * workers)
    return sum(rewards) / total_episodes