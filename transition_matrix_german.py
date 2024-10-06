import os
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader

# Define the valid German characters (lowercase for case-insensitivity)
VALID_CHARACTERS = set("einrtasdhucglmobfzwkpvüäößjxyq")

#read the pdf
def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "  # Adding a space between pages
    return text

# Filter the text to keep only valid German characters (case insensitive)
def filter_text(text):
    return ''.join(c for c in text if c in VALID_CHARACTERS)

# Compute the transition matrix from the given sequence of characters.
def compute_transition_matrix(sequence):
    states = sorted(set(sequence))
    n_states = len(states)
    transition_matrix = np.zeros((n_states, n_states))
    state_to_index = {state: index for index, state in enumerate(states)}

    for (current, next_) in zip(sequence[:-1], sequence[1:]):
        transition_matrix[state_to_index[current], state_to_index[next_]] += 1

    return transition_matrix, states

# Compute the cumulative transition matrix from all PDF files in the folder.
def compute_cumulative_transition_matrix(folder_path):
    cumulative_transition_matrix = None
    all_states = None

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'): #Ensure is pdf
            file_path = os.path.join(folder_path, filename)
            text = read_pdf(file_path)
            clean_text = filter_text(text.lower())
            transition_matrix, states = compute_transition_matrix(clean_text)

            if cumulative_transition_matrix is None:
                cumulative_transition_matrix = transition_matrix
                all_states = states
            else:
                if states != all_states:
                    all_states_set = set(all_states) | set(states)
                    new_states = sorted(all_states_set)
                    new_transition_matrix = np.zeros((len(new_states), len(new_states)))
                    new_state_to_index = {state: index for index, state in enumerate(new_states)}

                    for i in range(len(all_states)):
                        for j in range(len(all_states)):
                            new_transition_matrix[new_state_to_index[all_states[i]], new_state_to_index[all_states[j]]] += cumulative_transition_matrix[i, j]

                    for i in range(len(states)):
                        for j in range(len(states)):
                            new_transition_matrix[new_state_to_index[states[i]], new_state_to_index[states[j]]] += transition_matrix[i, j]

                    cumulative_transition_matrix = new_transition_matrix
                    all_states = new_states
                else:
                    cumulative_transition_matrix += transition_matrix

    # Normalize the transition matrix
    for i in range(len(all_states)):
        row_sum = cumulative_transition_matrix[i].sum()
        if row_sum > 0:
            cumulative_transition_matrix[i] /= row_sum

    # Create DataFrame for the transition matrix
    transition_df = pd.DataFrame(cumulative_transition_matrix, index=all_states, columns=all_states)
    return transition_df
