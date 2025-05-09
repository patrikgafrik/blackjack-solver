import numpy as np
import matplotlib.pyplot as plt
import random
import gymnasium as gym
env = gym.make('Blackjack-v1', natural=False, sab=False)
from concurrent.futures import ProcessPoolExecutor

# Define cellular automata parameters
class BlackjackCA:
    def __init__(self, num_generations=100, population_size=100, neighbourhood_size=2):
        self.num_generations = num_generations
        self.population_size = population_size
        self.neighbourhood_size = neighbourhood_size
        
        self.optimal_hard_totals = np.array([
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
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ])

        self.optimal_soft_totals = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ])

        # Initialize random strategy matrices
        self.hard_total_matrix = np.zeros((16, 10), dtype=int)
        self.soft_total_matrix = np.zeros((8, 10), dtype=int)
        
        # Initialize buffers for CA simulation
        self.hard_buffer = np.zeros((2, 16, 10), dtype=int)
        self.soft_buffer = np.zeros((2, 8, 10), dtype=int)
        
    def initialize_random_strategy(self):
        """Initialize random strategies for both hard and soft totals"""
        actions = [0, 1]  # 0 = Hit, 1 = Stand
        self.hard_total_matrix = np.random.choice(actions, size=(16, 10))
        self.soft_total_matrix = np.random.choice(actions, size=(8, 10))
        
    def get_cell_id(self, matrix, row, col):
        """Get cell value with boundary checking"""
        if row < 0:
            row = 0
        elif row >= matrix.shape[0]:
            row = matrix.shape[0] - 1
            
        if col < 0:
            col = 0
        elif col >= matrix.shape[1]:
            col = matrix.shape[1] - 1
            
        return row, col
    
    def apply_rule(self, matrix, rule_set, row, col):
        """Apply the cellular automaton rule to determine the next state"""
        rule_id = 0
        
        if self.neighbourhood_size == 1:
            # Consider 1 cell in each direction (3x3 neighborhood)
            neighbors = [
                matrix[self.get_cell_id(matrix, row-1, col)],  # North
                matrix[self.get_cell_id(matrix, row, col-1)],  # West
                matrix[row, col],                             # Center
                matrix[self.get_cell_id(matrix, row, col+1)],  # East
                matrix[self.get_cell_id(matrix, row+1, col)]   # South
            ]
            for i, val in enumerate(neighbors):
                rule_id |= (val << i)
                
        elif self.neighbourhood_size == 2:
            # Consider 2 cells in each direction (5x5 neighborhood)
            positions = [
                (row-2, col), (row-1, col), (row, col-2), (row, col-1),
                (row, col), (row, col+1), (row, col+2), (row+1, col), (row+2, col)
            ]
            for i, (r, c) in enumerate(positions):
                r, c = self.get_cell_id(matrix, r, c)
                rule_id |= (matrix[r, c] << i)
                
        return rule_set[rule_id % len(rule_set)]
    
    def simulate_step(self, matrix, rule_set, buffer):
        """Perform one step of the cellular automaton simulation using the provided buffer"""
        current = 0
        future = 1
    
        buffer[current] = matrix.copy()
    
        # Update each cell based on its neighborhood
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                buffer[future][row, col] = self.apply_rule(buffer[current], rule_set, row, col)
                
        return buffer[future].copy()
    
    def simulate(self, hard_matrix, soft_matrix, hard_rule_set, soft_rule_set, steps=10):
        """Run the cellular automaton for multiple steps on both matrices"""
        hard_result = hard_matrix.copy()
        soft_result = soft_matrix.copy()
    
        for _ in range(steps):
            hard_result = self.simulate_step(hard_result, hard_rule_set, self.hard_buffer)
            soft_result = self.simulate_step(soft_result, soft_rule_set, self.soft_buffer)
        
        return hard_result, soft_result
    
    def get_decision(self, strategy, player_sum, dealer_upcard, is_soft):
        """Get the action decision based on the strategy matrices"""
        hard_totals, soft_totals = strategy
        dealer_idx = dealer_upcard - 2  # Adjust dealer card index to match array indexing
    
        if is_soft == 1:
            row_index = max(0, min(20 - player_sum, soft_totals.shape[0] - 1))
            return soft_totals[row_index, dealer_idx]
        else:
            row_index = max(0, min(20 - player_sum, hard_totals.shape[0] - 1))
            return hard_totals[row_index, dealer_idx]
    
    # Evaluate a strategy by playing multiple games
    def evaluate_strategy(self, strategy, episodes=50000):
        """Evaluate a strategy by playing multiple games"""
        # Make sure strategy is a tuple of (hard_totals, soft_totals)
        hard_totals, soft_totals = strategy
        
        total_reward = 0
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                player_sum, dealer_card, usable_ace = obs
                if dealer_card == 1:
                    dealer_card = 11
                action = self.get_decision((hard_totals, soft_totals), player_sum, dealer_card, usable_ace)
                obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        return total_reward / episodes
    
    def optimize_strategy(self):
        """Use cellular automata to optimize the blackjack strategy"""
        best_hard_matrix = None
        best_soft_matrix = None
        best_fitness = -float('inf')
        
        # Generate random rule sets for the CA
        hard_rule_set = [random.randint(0, 1) for _ in range(2**(2*self.neighbourhood_size + 1))]
        soft_rule_set = [random.randint(0, 1) for _ in range(2**(2*self.neighbourhood_size + 1))]
        
        # Run the CA for multiple generations
        for generation in range(self.num_generations):
            # Apply CA rules to evolve the strategy
            new_hard_matrix, new_soft_matrix = self.simulate(
                self.hard_total_matrix, 
                self.soft_total_matrix, 
                hard_rule_set, 
                soft_rule_set, 
                steps=5
            )
            
            # Evaluate the new strategy
            fitness = self.evaluate_strategy((new_hard_matrix, new_soft_matrix))
            
            # Keep track of the best strategy
            if fitness > best_fitness:
                best_fitness = fitness
                best_hard_matrix = new_hard_matrix.copy()
                best_soft_matrix = new_soft_matrix.copy()
                
                # Calculate match percentages with optimal strategy
                hard_match_percentage = np.mean(new_hard_matrix == self.optimal_hard_totals) * 100
                soft_match_percentage = np.mean(new_soft_matrix == self.optimal_soft_totals) * 100
                
                print(f"Generation {generation}: New best fitness = {fitness:.4f}")
                print(f"Hard Totals Match Percentage: {hard_match_percentage:.2f}%")
                print(f"Soft Totals Match Percentage: {soft_match_percentage:.2f}%")
            
            # Update the current strategy with the new one
            self.hard_total_matrix = new_hard_matrix
            self.soft_total_matrix = new_soft_matrix
            
            # Sometimes mutate the rule set to explore new strategies
            if random.random() < 0.1:
                idx = random.randint(0, len(hard_rule_set) - 1)
                hard_rule_set[idx] = 1 - hard_rule_set[idx]  # Flip 0/1
                
                idx = random.randint(0, len(soft_rule_set) - 1)
                soft_rule_set[idx] = 1 - soft_rule_set[idx]  # Flip 0/1
                
        return best_hard_matrix, best_soft_matrix, best_fitness