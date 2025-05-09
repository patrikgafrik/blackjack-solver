import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Create the Blackjack environment
env = gym.make('Blackjack-v1')

class BlackjackNN:
    def __init__(self, state_size=3, action_size=2):
        self.state_size = state_size  # player sum, dealer card, usable ace
        self.action_size = action_size  # hit or stand
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=20000)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
    def _build_model(self):
        # Neural Net for Deep-Q learning
        model = keras.Sequential([
            layers.Dense(64, input_dim=self.state_size, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_network(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        # Store transition in memory
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        # Epsilon-greedy action selection
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state_tensor)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size):
        # Train on random minibatch from memory
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        # Current Q values of the sampled states
        targets = self.model.predict(states, verbose=0)
        
        # Future Q values from target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
                
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)


def normalize_state(state):
    # Convert the Blackjack state to a normalized numpy array
    player_sum, dealer_card, usable_ace = state
    
    # Normalize values between 0 and 1
    normalized_player_sum = (player_sum - 4) / 17.0  # player sum is between 4 and 21
    normalized_dealer_card = (dealer_card - 1) / 9.0  # dealer showing card is between 1 and 10
    normalized_usable_ace = 1.0 if usable_ace else 0.0
    
    return np.array([normalized_player_sum, normalized_dealer_card, normalized_usable_ace])


def train_blackjack_agent(episodes=20000, batch_size=64):
    """Train the neural network blackjack agent"""
    agent = BlackjackNN()
    
    rewards = []
    avg_rewards = []
    
    target_update_freq = 100  # Update target network every 100 episodes
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = normalize_state(state)
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done:
                next_state = np.zeros(agent.state_size)
            else:
                next_state = normalize_state(next_state)
                
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Replay experience
            agent.replay(batch_size)
            
        rewards.append(total_reward)
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
            
        # Calculate average reward over last 100 episodes
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            avg_rewards.append(avg_reward)
            print(f"Episode {episode}/{episodes}, Average Reward: {avg_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
    
    # Save the final model
    agent.save_model("blackjack_model.h5")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, episodes, 100), avg_rewards)
    plt.title('Average Reward over 100 episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.savefig('training_rewards.png')
    plt.show()
    
    return agent


def evaluate_agent(agent, num_games=10000):
    """Evaluate the trained agent"""
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(num_games):
        state, _ = env.reset()
        state = normalize_state(state)
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                if reward > 0:
                    wins += 1
                elif reward == 0:
                    draws += 1
                else:
                    losses += 1
                break
            state = normalize_state(next_state)
    
    print(f"Wins: {wins} ({wins/num_games:.4f})")
    print(f"Draws: {draws} ({draws/num_games:.4f})")
    print(f"Losses: {losses} ({losses/num_games:.4f})")
    
    return wins, draws, losses


def generate_strategy_table(agent):
    """Generate a strategy table from the trained agent"""
    strategy = np.zeros((18, 10, 2))  # player sum (4-21), dealer showing (1-10), usable ace (0-1)
    
    for player_sum in range(4, 22):
        for dealer_card in range(1, 11):
            for usable_ace in [False, True]:
                state = normalize_state((player_sum, dealer_card, usable_ace))
                action = agent.act(state, training=False)
                
                # Convert to 0-indexed
                player_idx = player_sum - 4
                dealer_idx = dealer_card - 1
                ace_idx = 1 if usable_ace else 0
                
                if player_idx < 18:  # We only track 4-21
                    strategy[player_idx, dealer_idx, ace_idx] = action
    
    return strategy


def plot_strategy(strategy):
    """Plot the learned strategy"""
    actions = ['Hit', 'Stand']
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    titles = ['No Usable Ace', 'Usable Ace']
    
    for ace in range(2):
        ax = axs[ace]
        data = strategy[:, :, ace]
        
        # Plot heatmap
        heatmap = ax.imshow(data, cmap='coolwarm', aspect='auto')
        
        # Add labels
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_title(titles[ace])
        
        # Add ticks
        ax.set_xticks(np.arange(10))
        ax.set_xticklabels([str(i) for i in range(1, 11)])
        ax.set_yticks(np.arange(18))
        ax.set_yticklabels([str(i) for i in range(4, 22)])
        
        # Add colorbar
        cbar = plt.colorbar(heatmap, ax=ax, ticks=[0, 1])
        cbar.set_ticklabels(['Hit', 'Stand'])
        
    plt.tight_layout()
    plt.savefig('strategy.png')
    plt.show()
    