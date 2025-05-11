import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import gymnasium as gym

# Create the Blackjack environment for evaluation
env = gym.make('Blackjack-v1' , natural=False, sab=False)


def create_training_data():
    """
    Create training data for the Blackjack neural network.
    
    Returns:
        tuple: (X, y) where X is the input features and y is the target actions
               1 = hit, 0 = stand
    """
    # Basic Blackjack strategy
    # Key: (player_sum, dealer_card, has_usable_ace)
    # Value: action (1 = hit, 0 = stand)
    
    X = []  # States: [player_sum, dealer_card, has_usable_ace]
    y = []  # Actions: 1 = hit, 0 = stand
    
    # Generate all possible states:
    # Player sum: 4-21
    # Dealer showing: 1-10 (where 1=Ace, 10=10/J/Q/K)
    # Usable ace: 0 or 1
    
    # Hard totals (no usable ace)
    for player_sum in range(4, 22):  # Valid player sums 4-21
        for dealer_card in range(1, 11):  # Dealer cards 1-10 (1=Ace)
            state = [(player_sum - 4) / 17.0, (dealer_card - 1) / 9.0, 0]  # No usable ace
            X.append(state)
            
            # Basic strategy for hard totals
            if player_sum >= 17:
                # Always stand on 17 or higher
                action = 0
            elif player_sum >= 13 and dealer_card < 7:
                # Stand on 13-16 against dealer 2-6
                action = 0
            elif player_sum == 12 and 4 <= dealer_card <= 6:
                # Stand on 12 against dealer 4-6
                action = 0
            else:
                # Otherwise hit
                action = 1
                
            y.append(action)
    
    # Soft totals (with usable ace)
    for player_sum in range(12, 22):  # Soft totals 12-21 (Ace + something)
        for dealer_card in range(1, 11):
            state = [(player_sum - 4) / 17.0, (dealer_card - 1) / 9.0, 1]
            X.append(state)
            
            # Basic strategy for soft totals
            if player_sum >= 19:
                # Always stand on soft 19 or higher
                action = 0
            elif player_sum == 18 and dealer_card < 9:
                # Stand on soft 18 against dealer 2-8
                action = 0
            else:
                # Otherwise hit
                action = 1
                
            y.append(action)
    
            
    return np.array(X), np.array(y)



class BlackjackNN:
    def __init__(self, layer_config=None):
        """
        Initialize the BlackjackNN with a customizable layer configuration.
        
        Args:
            layer_config (list of tuples): Each tuple specifies (units, activation) for a layer.
                                           Default is [(64, 'relu'), (32, 'relu')].
        """
        self.layer_config = layer_config or [(64, 'relu'), (32, 'relu')]
        self.model = self._build_model()
        self.x_train = None
        self.y_train = None
        
    def _build_model(self):
        """Build a neural network model based on the layer configuration."""
        model = keras.Sequential()
        model.add(layers.Input(shape=(3,)))  # Input layer
        
        # Add hidden layers based on the configuration
        for units, activation in self.layer_config:
            model.add(layers.Dense(units, activation=activation))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))  # Output: probability of standing (0) vs hitting (1)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=100, batch_size=16):
        """Train the model on optimal strategy data."""
        # Generate training data from optimal strategy matrices
        self.x_train, self.y_train = create_training_data()
        
        # Train the model
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        return history
    
    def save_model(self, filename="blackjack_supervised_model.h5"):
        """Save the trained model."""
        self.model.save(filename)
    
    def load_model(self, filename="blackjack_supervised_model.h5"):
        """Load a trained model."""
        self.model = keras.models.load_model(filename)
    
    def get_action(self, state):
        """Get the action for a specific state."""
        player_sum, dealer_card, usable_ace = state
        
        # Normalize the state
        normalized_state = np.array([
            [(player_sum - 4) / 17.0],
            [(dealer_card - 1) / 9.0],
            [1 if usable_ace == 1 else 0]
        ]).T
        
        # Predict action (sigmoid output > 0.5 -> hit, otherwise stand)
        prediction = self.model.predict(normalized_state, verbose=0)[0][0]
        return 1 if prediction > 0.5 else 0
    
    def evaluate(self, num_episodes=2000):
        """Evaluate the model by playing games."""
        total_reward = 0
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            
            while not done:
                action = self.get_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            total_reward += reward
    
        return total_reward / num_episodes