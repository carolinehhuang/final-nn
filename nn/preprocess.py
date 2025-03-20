# Imports
import numpy as np
import random
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    #zip the sequences and labels together to match the seq and labels up, then separate the sequences into positive and negative group
    g1_true = [seq for seq, label in zip(seqs, labels) if label]
    g2_false = [seq for seq, label in zip(seqs, labels) if not label]

    #find the number of samples in the larger imbalanced class
    balanced_class_n = max(len(g1_true), len(g2_false))

    #randomly sample the balanced_class_n elements from each set of sequences with replacement
    sampled_true = list(np.random.choice(g1_true, size=balanced_class_n, replace = True))
    sampled_false = list(np.random.choice(g2_false, size=balanced_class_n, replace = True))

    #put the dataset back together and create labels corresponding to these elements
    sampled_seqs = sampled_true + sampled_false
    new_labels = [1] * balanced_class_n + [0] * balanced_class_n

    #zip the sequences and the labels together and shuffle them together so the label and the sequence remain connected
    combined_seq_labels = list(zip(sampled_seqs, new_labels))
    np.random.shuffle(combined_seq_labels)

    #unzip the labels and the sequences and return as two separate lists
    random_sampled_seqs, random_labels = zip(*combined_seq_labels)

    return random_sampled_seqs, random_labels



def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    #create dictionary for mapping
    base_mapping = {'A': np.array([1,0,0,0]),
                    'T': np.array([0,1,0,0]),
                    'C': np.array([0,0,1,0]),
                    'G': np.array([0,0,0,1])}

    one_hot_encoded = []

    #for each sequence in the list, translate the basepair into its one hot encoding
    for seq in seq_arr:
        one_hot_seq = []
        for base in seq:
            one_hot_seq.append(base_mapping[base])
        one_hot_encoded.append(np.concatenate(np.array(one_hot_seq)))

    return np.array(one_hot_encoded)
