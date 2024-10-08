# Importing other documents
from transition_matrix_german import compute_cumulative_transition_matrix, read_pdf, filter_text
from entropy_german_actual_distribution import compute_entropy
import os

def compute_letter_probabilities(folder_path):
    letter_counts = {}
    total_count = 0

    # Look in all files
    for filename in os.listdir(folder_path): 
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            text = read_pdf(file_path)
            clean_text = filter_text(text.lower())  # Filter to valid characters

            # Count the frequency of each letter
            for char in clean_text:
                if char in letter_counts:
                    letter_counts[char] += 1
                else:
                    letter_counts[char] = 1
            
            total_count += len(clean_text)

    # Calculate probabilities
    letter_probabilities = {char: count / total_count for char, count in letter_counts.items()}

    return letter_probabilities

def main(folder_path):

    # Letter prob. of each file
    letter_probabilities = compute_letter_probabilities(folder_path)

    # Compute the cumulative transition matrix from all PDFs in the folder to get the joint probabilities
    transition_matrix = compute_cumulative_transition_matrix(folder_path)

    # Calculate the entropy using the transition matrix and letter probabilities
    entropy_value = compute_entropy(transition_matrix.to_numpy(), transition_matrix.index.tolist(), letter_probabilities)

    print(f"Final Cumulative Transition Matrix:\n{transition_matrix}\n")
    print(f"Entropy: {entropy_value:.4f} bits\n")

    # Print letter probabilities
    print("Letter Probabilities:")
    for letter, prob in letter_probabilities.items():
        print(f"{letter}: {prob:.4f}")

# Using the folder needed
folder_path = 'C:\\Users\\axeld\\Desktop\\Axel\\Developer\\Entropy-in-a-Language\\pdf_german_text' # Change this to your folder containing PDF files
if __name__ == "__main__":
    main(folder_path)
