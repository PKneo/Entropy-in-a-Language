# entropy_calculator.py
import numpy as np

def compute_entropy(transition_matrix, states, letter_probabilities):
    """Compute the entropy of a first-order Markov source."""
    # Ensure the transition matrix is a numpy array
    transition_matrix = np.array(transition_matrix)

    # Initialize entropy
    entropy = 0.0

    # Calculate entropy
    for i, state_i in enumerate(states):
        for j, state_j in enumerate(states):
            p_i = letter_probabilities.get(state_i, 0)  # Fixed probability of letter i
            p_ij = transition_matrix[i, j]  # Transition probability from i to j
            
            if p_ij > 0:  # Only consider positive probabilities
                entropy += p_i * p_ij * np.log2(p_ij)  # Use log base 2

    return -entropy  # The entropy should be negative of the sum

