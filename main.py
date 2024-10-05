# main.py
from confusion_matrix_german import compute_cumulative_transition_matrix
from entropy_germna_actual_distribution import compute_entropy

# Fixed probabilities for German letters
letter_probabilities = {
    'e': 0.1611,
    'n': 0.1033,
    'i': 0.0905,
    'r': 0.0672,
    't': 0.0634,
    's': 0.0623,
    'a': 0.0560,
    'h': 0.0520,
    'd': 0.0417,
    'u': 0.0370,
    'c': 0.0340,
    'l': 0.0324,
    'g': 0.0294,
    'm': 0.0280,
    'o': 0.0232,
    'b': 0.0219,
    'f': 0.0171,
    'w': 0.0139,
    'z': 0.0136,
    'k': 0.0133,
    'v': 0.0092,
    'p': 0.0084,
    'ü': 0.0064,
    'ä': 0.0051,
    'ö': 0.0036,
    'ß': 0.0019,
    'j': 0.0019,
    'x': 0.0011,
    'q': 0.0007,
    'y': 0.0006
}

def main(folder_path):
    # Compute the cumulative transition matrix from all PDFs in the folder
    transition_matrix = compute_cumulative_transition_matrix(folder_path)
    
    # Calculate the entropy using the transition matrix and letter probabilities
    entropy_value = compute_entropy(transition_matrix.to_numpy(), transition_matrix.index.tolist(), letter_probabilities)
    
    print(f"Final Cumulative Transition Matrix:\n{transition_matrix}\n")
    print(f"Entropy: {entropy_value:.4f} bits")

# Example usage
folder_path = 'D:\Developer\Python\Test\pdf_german_text'  # Change this to your folder containing PDF files
if __name__ == "__main__":
    main(folder_path)
