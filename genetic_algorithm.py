import gymnasium as gym
import numpy as np
env = gym.make('Blackjack-v1', natural=False, sab=False)
from concurrent.futures import ProcessPoolExecutor


class BlackjackGA:
    def __init__(self, population_size=100, mutation_rate=0.01, generations=100):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = [self.create_blackjack_matrices() for _ in range(population_size)]

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

    def create_blackjack_matrices(self):
        actions = [1, 0] # Hit, Stand
    
        # Hard totals (player total, dealer upcard: 2-10, A)
        hard_totals = np.array([[np.random.choice(actions) for _ in range(10)] for _ in range(16)])
    
        # Soft totals (player total with an Ace, dealer upcard: 2-10, A)
        soft_totals = np.array([[np.random.choice(actions) for _ in range(10)] for _ in range(8)])
    
        return (hard_totals, soft_totals)
        

    def get_decision(self, strategy, player_sum, dealer_upcard, is_soft):
        hard_totals, soft_totals = strategy
        dealer_idx = dealer_upcard - 2  # Adjust dealer card index to match array indexing
    
        if is_soft == 1:
            row_index = max(0, min(20 - player_sum, soft_totals.shape[0] - 1))
            return soft_totals[row_index, dealer_idx]
        else:
            row_index = max(0, min(20 - player_sum, hard_totals.shape[0] - 1))
            return hard_totals[row_index, dealer_idx]  # Adjusted index for hard_totals
        
    # Evaluate a strategy by playing multiple games
    def evaluate_strategy(self, strategy, episodes=50000):
        total_reward = 0
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                player_sum, dealer_card, usable_ace = obs
                if dealer_card == 1:
                    dealer_card = 11
                action = self.get_decision(strategy, player_sum, dealer_card, usable_ace)
                obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        return total_reward / episodes
    
    # Selection: Select the top-performing strategies
    def select_population(self, population, fitness, num_selected):
        selected_indices = np.argsort(fitness)[-num_selected:]
        selected_population = [population[i] for i in selected_indices]
        return np.array(selected_population, dtype=object)
    
    # Crossover: Combine two parent strategies to create a child strategy
    def crossover(self, parent1, parent2):
        # Use a new random seed each time
        mask = np.random.random(parent1.shape) < 0.5
        child = np.where(mask, parent1, parent2)
        return child
    
    # Mutation: Randomly modify parts of a strategy
    def mutate(self, strategy, mutation_rate):
        # Use a new random seed each time
        mutation_mask = np.random.random(strategy.shape) < mutation_rate
        strategy[mutation_mask] = 1 - strategy[mutation_mask]
        return strategy
            
    # Run the genetic algorithm
    def run(self):
        best_ever_fitness = float('-inf')
        best_ever_strategy = None
        elite_count = 2

        for generation in range(self.generations):
            # Evaluate the fitness of each strategy
            with ProcessPoolExecutor() as executor:
                fitness = list(executor.map(self.evaluate_strategy, self.population))
            
            # Find the best strategy in this generation
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            best_strategy = self.population[best_idx]
            
            # Track the best strategy ever found
            if best_fitness > best_ever_fitness:
                best_ever_fitness = best_fitness
                hard_totals, soft_totals = best_strategy
                hard_match_percentage = np.mean(hard_totals == self.optimal_hard_totals) * 100
                soft_match_percentage = np.mean(soft_totals == self.optimal_soft_totals) * 100
                print(f"Hard Totals Match Percentage: {hard_match_percentage:.2f}%")
                print(f"Soft Totals Match Percentage: {soft_match_percentage:.2f}%")
                best_ever_strategy = (hard_totals.copy(), soft_totals.copy())
            
            # Print the best fitness in the current generation
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
            
            # Select the top-performing strategies
            num_selected = len(self.population) // 2
            selected_population = self.select_population(self.population, fitness, num_selected)
            
            # Create the next generation (starting with elites)
            next_generation = []
            # Apply elitism - preserve the best strategies
            elite_indices = np.argsort(fitness)[-elite_count:]
            for idx in elite_indices:
                next_generation.append(self.population[idx])
            
            # Fill the rest with offspring
            while len(next_generation) < self.population_size:
                parent1_idx = np.random.randint(0, len(selected_population))
                parent2_idx = np.random.randint(0, len(selected_population))
                
                parent1, parent2 = selected_population[parent1_idx], selected_population[parent2_idx]
                
                # Create a deep copy to avoid modifying parents
                hard1, soft1 = parent1
                hard2, soft2 = parent2
                
                # Apply crossover separately to hard and soft totals
                child_hard = self.crossover(hard1, hard2)
                child_soft = self.crossover(soft1, soft2)
                
                # Apply mutation
                child_hard = self.mutate(child_hard, self.mutation_rate)
                child_soft = self.mutate(child_soft, self.mutation_rate)
                
                next_generation.append((child_hard, child_soft))
            
            self.population = next_generation

        return best_ever_strategy, best_ever_fitness